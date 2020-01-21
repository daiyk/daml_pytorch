import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import os
import sys
import numpy as np
from utils import Timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from torch_utils import *


class daml_sim(nn.Module):
    def __init__(self, dU=7, state_idx=None, img_idx=None, config_params=None):
        super(daml_sim, self).__init__()
        # save important params
        self.dU = dU  # dim output/action
        self.state_idx = state_idx
        self.img_idx = img_idx
        self.config = config_params
        self.dO = len(state_idx) + len(img_idx)  # dim of input = flatten img + state
        # build model
        self.gripper_pose_min = config_params["gripper_pose_min"]
        self.gripper_pose_max = config_params["gripper_pose_max"]
        # VGG CNN layers
        with Timer("Building pytorch network"):
            self.construct_model(self.dO, self.dU)

    # add 'training' or 'testoing' prefix latter
    def construct_model(self, dim_input=None, dim_output=None):
        # build model
        # VGG CNN layers
        num_filters = self.config.get("num_filters", 64)  # default 64 for pushing
        filter_size = self.config.get("filter_size", 3)  # default 3 for pushing

        # reshape the input image to 100*90
        im_height = self.config["image_height"]
        im_width = self.config["image_width"]
        num_channels = self.config["image_channels"]

        initialization = self.config.get(
            "initialization", "xavier"
        )  # weights initialization xavier for pushing, TODO: add more initialization option

        n_conv_layer = self.config[
            "n_conv_layer"
        ]  # hardcode conv layer num not used here

        self.n_conv_output = (
            num_filters * 2
        )  # hardcode for spatial softmax after 2d cnv, equals to the twice size of filters

        num_strides = self.config[
            "num_strides"
        ]  # n_strides = 4, daml pushing strides = 2 for all 4 conv layers

        fan_in = num_channels * 2  # input channel plus bias transformation channel

        self.relu = nn.ReLU()  # build relu
        self.softmax = nn.Softmax(dim=-1)  # build softmax for spatial softmax
        # init bias transformation, add to parameters and clip it to [0.0,1.0]
        self.conv_bt = nn.Parameter(
            torch.clamp(
                torch.zeros(
                    [num_channels, im_height, im_width],
                    dtype=torch.float32,
                    requires_grad=True,
                ),
                min=0.0,
                max=1.0,
            )
        )

        ###### 2d convolution layers total = 4 layers ????solve padding problem.???
        self.conv1 = nn.Conv2d(fan_in, num_filters, filter_size, stride=2)
        padding_shape_1 = [0, 1, 0, 1]  # output = [50，45,64]
        self.padding1 = zero_padding(padding_shape_1)

        # init weight and bias
        init_weights_xavier(self.conv1.weight)
        init_weights_zeros(self.conv1.bias)

        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, stride=2)
        padding_shape_2 = [0, 1, 1, 1]  # output = [25,23,64]
        self.padding2 = zero_padding(padding_shape_2)

        init_weights_xavier(self.conv2.weight)
        init_weights_zeros(self.conv2.bias)

        self.conv3 = nn.Conv2d(num_filters, num_filters, filter_size, stride=2)
        padding_shape_3 = [1, 1, 1, 1]  # output shape = [13,12,64]
        self.padding3 = zero_padding(padding_shape_3)
        init_weights_xavier(self.conv3.weight)
        init_weights_zeros(self.conv3.bias)

        self.conv4 = nn.Conv2d(num_filters, num_filters, filter_size, stride=1)
        padding_shape_4 = [1, 1, 1, 1]  # output = [13,12,64]
        self.padding4 = zero_padding(padding_shape_4)
        init_weights_xavier(self.conv4.weight)
        init_weights_zeros(self.conv4.bias)

        self.conv5 = nn.Conv2d(num_filters, num_filters, filter_size, stride=1)
        padding_shape_5 = [1, 1, 1, 1]  # output = [13,12,64]
        self.padding5 = zero_padding(padding_shape_5)
        init_weights_xavier(self.conv5.weight)
        init_weights_zeros(self.conv5.bias)

        ########## temporal convolution layers: predicted final eept, upper head ######
        ########## ??? the output state should be corresponding to the provided data ######
        temporal_filter_size = self.config.get(
            "temporal_filter_size", 20
        )  # tcn filter size
        n_temporal_filter = self.config.get("n_temporal_filter_final_eep", 20)
        temporal_shape_in = self.n_conv_output  # set to the size of conv output
        # tcn end effector pose
        self.tconv1_eep = nn.Conv1d(
            temporal_shape_in, n_temporal_filter, temporal_filter_size
        )
        padding_shape_eep1=[4,5]
        self.tpadding_eep1 = zero_padding(padding_shape_eep1,conv1d=True)
        
        init_weights_normal(self.tconv1_eep.weight, std=0.01)
        init_weights_zeros(self.tconv1_eep.bias)

        self.tconv2_eep = nn.Conv1d(
            n_temporal_filter, n_temporal_filter, temporal_filter_size
        )

        padding_shape_eep2=[4,5]
        self.tpadding_eep2 = zero_padding(padding_shape_eep2,conv1d=True)

        init_weights_normal(self.tconv2_eep.weight, std=0.01)
        init_weights_zeros(self.tconv2_eep.bias)

        # final output should be the predicted gripper pose
        self.tconv3_eep = nn.Conv1d(
            n_temporal_filter, self.gripper_pose_max - self.gripper_pose_min, 1
        )

        init_weights_normal(self.tconv3_eep.weight, std=0.01)
        init_weights_zeros(self.tconv3_eep.bias)

        ###### fully-connected layers preparation ######
        fc_shape_in = self.n_conv_output + (
            self.gripper_pose_max - self.gripper_pose_min
        )  # after spatial softmax 2*channel concat predicted gripper pose
        # concat state vector
        fc_shape_in += len(self.state_idx)
        # input concat bias transformation
        fc_shape_in += self.config.get("bt_dim", 20)
        # build weight for fc bias transformation
        self.fc_bt = nn.Parameter(
            torch.zeros(
                self.config.get("bt_dim", 20), dtype=torch.float32, requires_grad=True
            )
        )

        ####### build fully-connected layers #######
        n_fc_layer = self.config.get("n_fc_layer", 4)  # default 3 for pushing daml
        fc_layer_size = self.config.get(
            "fc_layer_size", 50
        )  # default 50 for pushing daml

        self.fc1 = nn.Linear(fc_shape_in, fc_layer_size)
        init_weights_normal(self.fc1.weight, std=0.01)
        init_weights_zeros(self.fc1.bias)

        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)

        init_weights_normal(self.fc2.weight, std=0.01)
        init_weights_zeros(self.fc2.bias)

        self.fc3 = nn.Linear(
            fc_layer_size, fc_layer_size
        )  # ?? double the use of this layer, output is action
        init_weights_normal(self.fc3.weight, std=0.01)
        init_weights_zeros(self.fc3.bias)

        self.fc4 = nn.Linear(
            fc_layer_size, self.dU
        )  # ?? double the use of this layer, output is action
        init_weights_normal(self.fc4.weight, std=0.01)
        init_weights_zeros(self.fc4.bias)

        ########## fc_layer finished

        ########## temporal convolution layers: adaptation loss, upper head
        temporal_filter_size = self.config.get(
            "temporal_filter_size", 10
        )  # tcn filter size
        n_temporal_filter = self.config.get("n_temporal_filter", 10)
        temporal_in_shape = self.config.get(
            "fc_layer_size", 50
        )  # set to the size of final hidden layer size 50
        # tcn
        self.tconv1 = nn.Conv1d(
            temporal_in_shape, n_temporal_filter, temporal_filter_size
        )
        tpadding_shape1 = [4,5]
        self.tpadding1 = zero_padding(tpadding_shape1,conv1d=True)
        init_weights_normal(self.tconv1.weight, std=0.01)
        init_weights_zeros(self.tconv1.bias)

        self.tconv2 = nn.Conv1d(
            n_temporal_filter, n_temporal_filter, temporal_filter_size
        )
        tpadding_shape2 = [4,5]
        self.tpadding2 = zero_padding(tpadding_shape2,conv1d=True)
        init_weights_normal(self.tconv2.weight, std=0.01)
        init_weights_zeros(self.tconv2.bias)

        self.tconv3 = nn.Conv1d(n_temporal_filter, self.dU, 1)

        init_weights_normal(self.tconv3.weight, std=0.01)
        init_weights_zeros(self.tconv3.bias)

    def forward_conv(self, image_input, testing=False):
        img_width = self.config["image_width"]
        img_height = self.config["image_height"]
        img_channel = self.config["image_channels"]

        # get decay
        decay = self.config.get("decay", 0.9)

        downsample_factor = self.config.get("stride", 2)

        # build bias transformation for conv2d
        image_input = image_input
        bt_conv = torch.zeros_like(image_input, dtype=torch.float32)
        bt_conv += self.conv_bt
        # concat image input and bias transformation
        conv_out = torch.cat((image_input, bt_conv), 1)

        ########## start CNN forward
        conv_out = self.padding1(conv_out)
        conv_out = self.conv1(conv_out)
        ###TODO change to custom layer_norm
        conv_out = self.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.padding2(conv_out)
        conv_out = self.conv2(conv_out)
        ###TODO change to custom layer_norm
        conv_out = self.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.padding3(conv_out)
        conv_out = self.conv3(conv_out)
        ###TODO change to custom layer_norm
        conv_out = self.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.padding4(conv_out)
        conv_out = self.conv4(conv_out)
        ###TODO change to custom layer_norm
        conv_out = self.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        conv_out = self.padding5(conv_out)
        conv_out = self.conv5(conv_out)  # final output is [B,64,13,12]
        ###TODO change to custom layer_norm
        conv_out = self.relu(F.layer_norm(conv_out, conv_out.shape[1:]))

        # spatial softmax layers
        _, fp, rows, cols = conv_out.shape  # [B*T,64,13,12]
        x_map = torch.zeros((rows, cols), dtype=torch.float32)
        y_map = torch.zeros((rows, cols), dtype=torch.float32)

        for i in range(rows):
            for j in range(cols):
                x_map[i, j] = (i - rows / 2.0) / rows
                y_map[i, j] = (j - cols / 2.0) / cols
        x_map = x_map.view(cols * rows, -1)
        y_map = y_map.view(cols * rows, -1)

        # reshape the conv_out to [N*C,H*W]
        feature_pts = conv_out.view(-1, cols * rows)  # [B*T*fp,13*12]

        feature_af_softmax = self.softmax(
            feature_pts
        )  # output = [T*B*fp,64] [B*T*fp,13*12]

        fp_x = torch.sum(
            torch.matmul(feature_af_softmax, x_map), dim=1, keepdim=True
        )  # [num_fp * T*B,1]
        fp_y = torch.sum(
            torch.matmul(feature_af_softmax, y_map), dim=1, keepdim=True
        )  # [num_fp * T*B,1]

        # get output flat features
        conv_out_features = torch.cat((fp_x, fp_y), dim=1).view(
            -1, fp * 2
        )  # #[ T*B, 2 * num_fp]=[B*T,128]

        return conv_out_features  # the expected 2d position

    # build forward network for fully connected layer
    def forward_fc(self, conv_out, state_input, meta_testing=False, testing=False):
        # build bias transformation for the fc layer

        conv_out_size = conv_out.shape[-1]
        fc_input = torch.add(conv_out, 0)

        eep_input = torch.reshape(
            conv_out, [-1, conv_out_size, self.config["frames"]]
        )  # [B,128,T]

        # final_ee_input = eep_input[:,0,:]
        # ?? temporal structure need to be adjusted to fit the output shape as [B, T, gripper_pose]
        eep_input = self.tpadding_eep1(eep_input)
        eep_output = self.tconv1_eep(eep_input)
        eep_output = self.relu(F.layer_norm(eep_output, eep_output.shape[1:]))

        # print("fc_output1: {}\n".format(fc_output.shape))
        eep_output = self.tpadding_eep2(eep_output)
        eep_output = self.tconv2_eep(eep_output)
        eep_output = self.relu(F.layer_norm(eep_output, eep_output.shape[1:]))

        # print("fc_output2: {}\n".format(fc_output.shape))
        eep_output = self.tconv3_eep(eep_output)  # [B,3,frames]

        # reshape and concat to the fc_input
        eep_output = torch.reshape(
            (torch.reshape(eep_output, [-1])).repeat(self.config["frames"]),
            [-1, self.gripper_pose_max - self.gripper_pose_min],
        )

        # output = [B, conv_output,gripper] -> [T*conv_out, gripper_state]
        batch_num = conv_out.shape[0]
        bt_fc = torch.zeros(
            batch_num, self.config.get("bt_dim", 20)
        )  # build container for fc bias transf
        # build fc bias transformation weights->[batch,weights]
        bt_fc += self.fc_bt
        # concat input vectors
        fc_input = torch.cat([conv_out, bt_fc], dim=1)

        # concat input with eep vector
        fc_input = torch.cat([fc_input, eep_output], dim=1) # 151
        # print("fc_input: {} state_input: {}".format(fc_input.shape, state_input.shape))
        fc_input = torch.cat([fc_input, state_input], dim=1)

        # go through linear layers
        fc_output = self.relu(self.fc1(fc_input))
        fc_output = self.relu(self.fc2(fc_output))
        fc_output = self.relu(self.fc3(fc_output))
        fc_output = self.fc4(fc_output) # final action output

        # TODO concatenate feature points with fc output
        fc_output=torch.cat([fc_output,conv_out],dim=1)

        # temporal CNN only works on the last hidden layer and only for inner update
        if not meta_testing:
            # temporal conv adaptation loss
            fc_output = fc_output.view(
                -1, fc_output.shape[-1], self.config.get("frames", 100)
            )  # batch*
            # go through temporal convolution
            # print("fc_input: {}\n".format(fc_output.shape))
            fc_output=self.tpadding1(fc_output)
            fc_output = self.tconv1(fc_output)
            fc_output = self.relu(F.layer_norm(fc_output, fc_output.shape[1:]))
            # print("fc_output1: {}\n".format(fc_output.shape))

            fc_output=self.tpadding2(fc_output)
            fc_output = self.tconv2(fc_output)
            fc_output = self.relu(F.layer_norm(fc_output, fc_output.shape[1:]))
            # print("fc_output2: {}\n".format(fc_output.shape))
            fc_output = self.tconv3(fc_output)
            return fc_output
        else:
            # lower head output
            fc_output = self.fc4(fc_output)
            return fc_output,eep_output

    def preprocess_input(self, input, state_idx, img_idx):
        
        #input data=states+image
        input_states = input[:, : state_idx[-1] + 1]
        input_flatten_images = input[:, state_idx[-1] + 1 : img_idx[-1] + 1]

        # transform back to [w,h,c]
        im_height = self.config.get("image_height")
        im_width = self.config.get("image_width")
        im_channels = self.config.get("image_channels")

        # default to transform it to NCHW
        input_images = input_flatten_images.reshape(
            -1, im_channels, im_height, im_width
        )

        return input_images, input_flatten_images, input_states

    def meta_learner(self, input, testing=False):
        """ two head meta-training and meta-learning function
            return meta-testing loss
        """
        # input is two input data: image1+state1 and image2+state2
        # two label data: actiona and actionb
        input1, input2, action1, action2 = input
        if type(input1).__module__ == np.__name__:
            input1 = torch.from_numpy(input1).view(-1, self.dO)
            input2 = torch.from_numpy(input2).view(-1, self.dO)
            action1 = torch.from_numpys(action1).view(-1, self.dU)
            action2 = torch.from_numpy(action2).view(-1, self.dU)
        else:
            input1 = input1.view(-1, self.dO)
            input2 = input2.view(-1, self.dO)
            action1 = action1.reshape(-1, self.dU)
            action2 = action2.reshape(-1, self.dU)

        # TODO ???need to consider the final gripper pose loss, need the index for gripper pose (end-effector pose) in your state vector
        # final eept1 is not used for preupdate
        final_eep1 = action1[:,self.gripper_pose_min:self.gripper_pose_max+1]
        final_eep2 = action2[:,self.gripper_pose_min:self.gripper_pose_max+1]
        action1 = action1[:,:self.gripper_pose_min]
        action2 = action2[:,self.gripper_pose_min]

        inner_train_update_lr = self.config.get("inner_train_update_lr", 0.01)
        total_loss2 = []  # total meta-testing loss
        total_eep_loss = []
        total_output2 = []  # store post-update resultant action data
        stored_weights = {}  # store weights for restore later

        if self.config.get("zero_action", True):
            action1 = torch.zeros_like(
                action1, dtype=torch.float32
            )  # zero out action for inner update

        input_images1, _, input_states1 = self.preprocess_input(
            input1, self.state_idx, self.img_idx
        )
        input_images2, input_flatten_images2, input_states2 = self.preprocess_input(
            input2, self.state_idx, self.img_idx
        )
        if self.config.get("zero_states", True):
            input_states1 = torch.zeros_like(
                input_states1, dtype=torch.float32
            )  # zero out states for
        # get number of inner gradient update
        num_updates = self.config.get("num_updates", 1)
        input_images1 = [input_images1] * num_updates
        # input_images2 = [input_images2] * num_updates

        action1 = [action1] * num_updates
        input_states1 = [input_states1] * num_updates

        ### pre-update forward network
        conv_out1 = self.forward_conv(input_images1[0], testing=testing)
        output1,eep_output = self.forward_fc(conv_out1, input_states1[0], testing=testing)

        # change it if for consider action input
        local_loss1 = self.config.get("action_loss_coeff", 1.0) * euclidean_loss(
            output1,
            torch.zeros_like(output1),
            multiplier=self.config.get("loss_multiplier", 50),
        )
       
        preupdate_loss = local_loss1
        # zero gradient before backward
        self.zero_grad()
        local_loss1.backward()
        # clip gradient value
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.config.get("clip_max", 10)
        )
        # hard-code gradient update
        for name, param in self.named_parameters():
            stored_weights[name] = param.data  # store param data
            if not param.grad is None:
                param.data.sub_(param.grad.data * inner_train_update_lr)

        #### post-update and return output and loss
        conv_out2 = self.forward_conv(input_images2, testing=testing)
        output2 = self.forward_fc(
            conv_out2, input_states2, meta_testing=True, testing=testing
        )
        total_output2.append(output2)
        local_loss2 = self.config.get("action_loss_coeff", 1.0) * euclidean_loss(
            output2, action2, multiplier=self.config.get("loss_multiplier", 50)
        )
        eep_loss = euclidean_loss(eep_output,final_eep2,multiplier=self.config.get("loss_multiplier", 50))
        local_loss2=eep_loss+local_loss2
        total_loss2.append(local_loss2)


        #### repeat for num_updates for gradient update
        # clip gradient value
        for j in range(num_updates - 1):
            ### pre-update forward network
            conv_out1 = self.forward_conv(input_images1[j + 1], testing=testing)
            output1,eep_output = self.forward_fc(conv_out1, input_states1[j + 1], testing=testing)
            local_loss1 = self.config.get("action_loss_coeff", 1.0) * euclidean_loss(
                output1,
                torch.zeros_like(output1),
                multiplier=self.config.get("loss_multiplier", 50),
            )

            # zero gradient before backward
            self.zero_grad()
            local_loss1.backward()
            # clip gradient value
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.get("clip_max", 10)
            )
            # hard-code gradient update
            for f in self.parameters():
                f.data.sub_(f.grad.data * inner_train_update_lr)

            #### post-update and return output and loss
            conv_out2 = self.forward_conv(input_images2, testing=testing)
            output2 = self.forward_fc(
                conv_out2, input_states2, meta_testing=True, testing=testing
            )
            total_output2.append(output2)
            local_loss2 = self.config.get("action_loss_coeff", 1.0) * euclidean_loss(
                output2, action2, multiplier=self.config.get("loss_multiplier", 50),
            )
            eep_loss = euclidean_loss(eep_output,final_eep2,multiplier=self.config.get("loss_multiplier", 50))
            local_loss2=eep_loss+local_loss2
            total_loss2.append(local_loss2)
        # recover the weights data
        for name, param in self.named_parameters():
            param.data = stored_weights[name]  # restore weights

        ## return actions and loss
        return [preupdate_loss, total_output2, total_loss2]

