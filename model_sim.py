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
        # dim of input = flatten img + state
        self.dO = len(state_idx) + len(img_idx)

        # build model
        # VGG CNN layers
        with Timer("Building pytorch network"):
            self.construct_model(self.dO, self.dU)

    # add 'training' or 'testoing' prefix latter
    def construct_model(self, dim_input=None, dim_output=None):
        # build model
        # VGG CNN layers
        num_filters = self.config.get("num_filters", 16)  # default 16 for pushing
        filter_size = self.config.get("filter_size", 5)  # default 5 for pushing

        im_height = self.config["image_height"]
        im_width = self.config["image_width"]
        num_channels = self.config["image_channels"]

        initialization = self.config.get(
            "initialization", "xavier"
        )  # weights initialization xavier for pushing

        n_conv_layer = self.config["n_conv_layer"]  # hardcode conv layer num

        self.n_conv_output = num_filters * 2  # hardcode for spatial softmax after 2d cnv, equals to the twice size of filters 

        num_strides = self.config["num_strides"] # n_strides = 4, daml pushing strides = 2 for all 4 conv layers

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

        ###### 2d convolution layers total = 4 layers ????1. solve padding problem. 2. stride_num = 4 and conv_layers = 4 not accordance with paper 5 layers and 3 strides???
        ###???layernorm need to modify with custom beta and gama
        self.conv1 = nn.Conv2d(fan_in, num_filters, filter_size, stride=2)
        padding_shape_1 = 2  # output = [63,63]
        self.padding1 = zero_padding(padding_shape_1)

        # init weight and bias
        init_weights_xavier(self.conv1.weight)
        init_weights_zeros(self.conv1.bias)

        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, stride=2)
        padding_shape_2 = 2  # output = [32,32]
        self.padding2 = zero_padding(padding_shape_2)
        init_weights_xavier(self.conv2.weight)
        init_weights_zeros(self.conv2.bias)

        self.conv3 = nn.Conv2d(num_filters, num_filters, filter_size, stride=2)
        padding_shape_3 = [1, 2, 1, 2]  # output shape = [16,16]
        self.padding3 = zero_padding(padding_shape_3)
        init_weights_xavier(self.conv3.weight)
        init_weights_zeros(self.conv3.bias)

        self.conv4 = nn.Conv2d(num_filters, num_filters, filter_size, stride=2)
        padding_shape_4 = [1, 2, 1, 2]  # output = [8,8]
        self.padding4 = zero_padding(padding_shape_4)
        init_weights_xavier(self.conv4.weight)
        init_weights_zeros(self.conv4.bias)

        ###### fully-connected layers
        fc_shape_in = self.n_conv_output  # after spatial softmax 2*channel
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

        ####### build fully-connected layer#######
        n_fc_layer = self.config.get("n_fc_layer", 3)  # default 3 for pushing daml
        fc_layer_size = self.config.get(
            "fc_layer_size", 200
        )  # default 200 for pushing daml

        self.fc1 = nn.Linear(fc_shape_in, fc_layer_size)
        init_weights_normal(self.fc1.weight, std=0.01)
        init_weights_zeros(self.fc1.bias)

        self.fc2 = nn.Linear(fc_layer_size, fc_layer_size)

        init_weights_normal(self.fc2.weight, std=0.01)
        init_weights_zeros(self.fc2.bias)

        self.fc3 = nn.Linear(
            fc_layer_size, self.dU
        )  # ?? double the use of this layer, output is action

        init_weights_normal(self.fc3.weight, std=0.01)
        init_weights_zeros(self.fc3.bias)

        ########## fc_layer finished
        ########## temporal convolution layers: adaptation loss, upper head
        temporal_filter_size = self.config.get(
            "temporal_filter_size", 10
        )  # tcn filter size
        n_temporal_filter = self.config.get("n_temporal_filter", 32)
        temporal_in_shape = self.config.get(
            "fc_layer_size", 100
        )  # set to the size of hidden layer size
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
        tpadding_shape2=[4,5]
        self.tpadding2=zero_padding(tpadding_shape2,conv1d=True)
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

        # spatial softmax layers
        _, fp, rows, cols = conv_out.shape
        x_map = torch.zeros((rows, cols), dtype=torch.float32)
        y_map = torch.zeros((rows, cols), dtype=torch.float32)

        for i in range(rows):
            for j in range(cols):
                x_map[i, j] = (i - rows / 2.0) / rows
                y_map[i, j] = (j - cols / 2.0) / cols
        x_map = x_map.view(cols * rows, -1)
        y_map = y_map.view(cols * rows, -1)

        # reshape the conv_out to [N*C,H*W]
        feature_pts = conv_out.view(-1, cols * rows)

        feature_af_softmax = self.softmax(feature_pts)  # outpur = [T*B,64]

        fp_x = torch.sum(
            torch.matmul(feature_af_softmax, x_map), dim=1, keepdim=True
        )  # [num_fp * T*B,1]
        fp_y = torch.sum(
            torch.matmul(feature_af_softmax, y_map), dim=1, keepdim=True
        )  # [num_fp * T*B,1]

        # get output flat features
        conv_out_features = torch.cat((fp_x, fp_y), dim=1).view(
            -1, fp * 2
        )  # #[ T*B, 2 * num_fp]

        return conv_out_features

    # build forward network for fully connected layer
    def forward_fc(self, conv_out, state_input, meta_testing=False, testing=False):
        # build bias transformation for the fc layer
        batch_num = conv_out.shape[0]
        bt_fc = torch.zeros(
            batch_num, self.config.get("bt_dim", 20)
        )  # build container for fc bias transf
        # build fc bias transformation weights->[batch,weights]
        bt_fc += self.fc_bt
        # concat input vectors
        fc_input = torch.cat([conv_out, bt_fc], dim=1)
        # print("fc_input: {} state_input: {}".format(fc_input.shape, state_input.shape))
        fc_input = torch.cat([fc_input, state_input], dim=1)

        # go through linear layers
        fc_output = self.relu(self.fc1(fc_input))
        fc_output = self.relu(self.fc2(fc_output))

        # temporal CNN only works on the last hidden layer
        if not meta_testing:
            # temporal conv adaptation loss
            # ???temporal convolution no relu and layernorm except final not accordanc with paper???
            fc_output = fc_output.view(
                -1, fc_output.shape[-1], self.config.get("frames", 100)
            )  # batch*
            # go through temporal convolution
            # print("fc_input: {}\n".format(fc_output.shape))
            self.tpadding1(fc_output)
            fc_output = self.tconv1(fc_output)
            self.tpadding2(fc_output)
            fc_output = self.tconv2(fc_output)
            # print("fc_output2: {}\n".format(fc_output.shape))
            fc_output = self.tconv3(fc_output)
            fc_output = self.relu(F.layer_norm(fc_output, fc_output.shape[1:]))
            return fc_output
        else:
            # ???????? no relu ????????
            # lower head output
            fc_output = self.fc3(fc_output)
            return fc_output

    def preprocess_input(self, input, state_idx, img_idx):
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
        inner_train_update_lr = self.config.get("inner_train_update_lr", 0.01)
        total_loss2 = []  # total meta-testing loss
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
        output1 = self.forward_fc(conv_out1, input_states1[0], testing=testing)

        #!!!
        # output1 = (output1.transpose(1, 2)).reshape(-1, self.dU)
        # output1 = output1[: action1[0].shape[0], :]
        #!!!

        # print("output; {}\n action: {}".format(output1.shape, action1[0].shape))

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
        clip_grad_value_(self.parameters(), self.config.get("clip_max", 10))
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
            output2, action2[0], multiplier=self.config.get("loss_multiplier", 50),
        )
        total_loss2.append(local_loss2)

        #### repeat for num_updates for gradient update
        # clip gradient value
        for j in range(num_updates - 1):
            ### pre-update forward network
            conv_out1 = self.forward_conv(input_images1[j + 1], testing=testing)
            output1 = self.forward_fc(conv_out1, input_states1[j + 1], testing=testing)
            local_loss1 = self.config.get("action_loss_coeff", 1.0) * euclidean_loss(
                output1,
                torch.zeros_like(output1),
                multiplier=self.config.get("loss_multiplier", 50),
            )

            # zero gradient before backward
            self.zero_grad()
            local_loss1.backward()
            # clip gradient value
            clip_grad_value_(self.parameters(), self.config.get("clip_max", 10))
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
            total_loss2.append(local_loss2)
        # recover the weights data
        for name, param in self.named_parameters():
            param.data = stored_weights[name]  # restore weights

        ## return actions and loss
        return [preupdate_loss, total_output2, total_loss2]

