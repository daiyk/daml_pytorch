import os
import sys
import torch
import torch.optim as optim
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from dataloader import *

# other parameters not defined but can be added:
#   "action_loss_coeff": default 1.0, action loss multiplier
#   "loss_multiplier": default 50, factor that use to scale the euclidean loss
#   "stop_grad": default false, if True, do not use second derivatives in meta-optimization (for speed)"
#   "clip": default true, whether to do gradient cliping
#   "zero_states": default true, whether states is required for inner training gradient update
#   "zero_action": default true, whether delete groundtruth of action during inner update, true for temporal adapation loss compute
#   "update_batch_size": default 1,  number of demo video for inner gradient update, k for k-shot learning (k videos)
#   "test_batch_size": default 1, number of examples used for valid and thus post-update gradient descend
# action=pos+vel+gripper_pos=3+3+3=9
# input = states+image
model_file = "model_pushing"

if model_file=="model_sim":
    network_config = {
        "model": "model_sim", # choose model, default model_sim
        "demo_file": "data/sim_push",  # path to demo files
        "demo_gif_dir": "data/sim_push",  # path to demo gif files
        "metatrain_iterations": 30000,  # metatrain iterations
        "val_set_size": 76,  # validation set size
        "image_width": 100,
        "image_height": 90,
        "image_channels": 3,
        "dim_output": 7,  # dimension of output action
        "meta_batch_size": 15,  # number of tasks sampled per meta-update
        "n_fc_layers": 3,  # number of fc layer, 3 for pushing daml
        "fc_layer_size": 200,  # fc layer size, 200 for pushing daml
        "temporal_filter_size": 10,  # temporal default to 10*10 for pushing daml
        "n_temporal_layer": 3,  # num of tcn layers,default pushing daml
        "n_temporal_filter": 32,  # temporal default to 32 for pushing daml
        "n_conv_layers": 4,  # num of conv layers, default pushing daml
        "filter_size": 5,  # conv2d filter pushing:5
        "bt_dim": 20,  # fc layer bias transformation dimension default pushing 20
        "num_filters": 16,  # conv2d num filters placing 64 and pushing 16, different from paper
        "initialization": "xavier",  # conv2d init default to xavier for pushing daml
        "num_updates": 1,  # number of inner gradient update step
        "loss_multiplier": 50.0,  # loss scaler value
        "inner_train_update_lr": 0.01,  # step size alpha for inner gradient update
        "outter_meta_lr": 0.001,  # learning rate of meta learning
        "decay": 0.9,  # batch norm decay, need to specify if used
        "stride": 2,  # stride number
        "frames": 100,  # number of frame of demo video
        "clip_min": -10,  # gradient cliping min
        "clip_max": 10,  # gradient cliping 
        "num_strides": 4, # number of stride layer, hardcoded not used for now
        "n_conv_layer": 4, # number of conv layers, hardcoded not used for now
    }
elif model_file == "model_pushing":
    network_config = {
        "model": "model_pushing", # choose model, default model_sim
        "demo_file": "data/sim_push",  # path to demo files
        "demo_gif_dir": "data/sim_push",  # path to demo gif files
        "metatrain_iterations": 30000,  # metatrain iterations
        "image_width": 100,
        "image_height": 90,
        "image_channels": 3,
        "dim_output": 7,  # dimension of output action
        "meta_batch_size": 15,  # number of tasks sampled per meta-update
        "n_fc_layers": 4,  # number of fc layer, 4 for pushing daml
        "fc_layer_size": 50,  # fc layer size, 50 for pushing daml
        "temporal_filter_size": 10,  # temporal default to 10*10 for action, pushing daml
        "n_temporal_layer": 3,  # num of tcn layers, 3 for action and gripper position, pushing daml
        "n_temporal_filter": 10,  # temporal filters default to 10X1 for action pushing daml
        "n_temporal_filter_final_eep": 20, # around half of the frames
        "n_conv_layers": 4,  # num of conv layers, default pushing daml with 5
        "filter_size": 3,  # conv2d filter size default to 3X3, for daml pushing
        "num_filters": 64,  # conv2d num filters placing 64 and pushing 64, different from paper
        "bt_dim": 20,  # fc layer bias transformation dimension, half of frames
        "initialization": "xavier",  # conv2d init default to xavier for pushing daml
        "num_updates": 5,  # number of inner gradient update step
        "loss_multiplier": 50.0,  # loss scaler value
        "inner_train_update_lr": 0.005,  # step size alpha for inner gradient update
        "outter_meta_lr": 0.001,  # learning rate of meta learning
        "decay": 0.9,  # batch norm decay, need to specify if used
        "frames": 40,  # number of frame of demo video
        "clip_min": -30,  # gradient cliping min
        "clip_max": 30,  # gradient cliping 
        "num_strides": 3, # number of stride layer, total 5 conv layers, hardcoded not used for now
        "stride": 2,  # stride number
        "gripper_pose_min": 0, # ??? first index of end effector pose in the action array, change later
        "gripper_pose_max": 2, # ??? last index of end effector pose in the action array, change later
        "val_set_size": 76,  # validation set size
        "learn_full_final_eep": True, # learn the full trajactory of eep
    }

WHC = (
    network_config["image_width"]
    * network_config["image_height"]
    * network_config["image_channels"]
)
MODEL=importlib.import_module(network_config["model"])
DIM_OUTPUT = network_config.get("dim_output")
META_BATH_SIZE = network_config.get("meta_batch_size", 15)
NUM_UPDATES = network_config.get("num_updates", 1)
META_LR = network_config.get("outter_meta_lr", 0.001)
ITERS = network_config.get("metatrain_iterations", 10000)
UPDATE_BATCH_SIZE = network_config.get("update_batch_szie", 1)
FRAMES = network_config.get("frames", 100)
datagenerator = DataGenerator(config=network_config)


def train(dataload, model):
    # define training object [meta_batch_size,(train=1+test+1)*frames,H*W*C]
    # [meta_batch_size,(train=1+test+1)*frames,states]
    # [meta_batch_size,(train=1+test+1)*frames,action]
    # dummy data
    dataloader = DataLoader(
        dataload,
        batch_size=network_config.get("meta_batch_size", 15),
        shuffle=False,
        num_workers=2,
    )

    optimizer = optim.Adam(model.parameters(), lr=META_LR)

    for i_batch, sampled_batch in enumerate(dataloader):
        image1 = sampled_batch[:, : FRAMES * UPDATE_BATCH_SIZE, :]
        image2 = sampled_batch[:, FRAMES * UPDATE_BATCH_SIZE :, :]

        state_batch, action_batch = datagenerator.generate_data_batch(i_batch)
        action1 = torch.from_numpy(action_batch[:, : FRAMES * UPDATE_BATCH_SIZE, :])
        action2 = torch.from_numpy(action_batch[:, FRAMES * UPDATE_BATCH_SIZE :, :])

        state1 = torch.from_numpy(state_batch[:, : FRAMES * UPDATE_BATCH_SIZE, :])
        state2 = torch.from_numpy(state_batch[:, FRAMES * UPDATE_BATCH_SIZE :, :])

        input1 = torch.cat([state1, image1], dim=2)
        input2 = torch.cat([state2, image2], dim=2)

        # train the model
        _, _, total_loss = model.meta_learner(
            [input1.float(), input2.float(), action1.float(), action2.float()]
        )
        loss_val = total_loss[-1]
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        #update via optimizer
        print("total loss: {}".format(total_loss))

    # image_plus_stats1 = torch.randint(
    #     low=0,
    #     high=255,
    #     size=(
    #         network_config["meta_batch_size"],
    #         network_config["frames"],
    #         WHC + len(state_idx),
    #     ),
    # )
    # image_plus_stats2 = torch.randint(
    #     low=0,
    #     high=255,
    #     size=(
    #         network_config["meta_batch_size"],
    #         network_config["frames"],
    #         WHC + len(state_idx),
    #     ),
    # )
    # image_plus_stats1 = image_plus_stats1.float()
    # image_plus_stats2 = image_plus_stats2.float()
    # action_1 = torch.randn(
    #     (network_config["meta_batch_size"], network_config["frames"], DIM_OUTPUT)
    # )
    # action_2 = torch.randn(
    #     (network_config["meta_batch_size"], network_config["frames"], DIM_OUTPUT)
    # )

    # #### test preprocess data function #####
    # image_plus_stats1 = image_plus_stats1.view(-1, (len(state_idx) + len(img_idx)))

    # output1, output2, output3 = model.preprocess_input(
    #     image_plus_stats1, state_idx, img_idx
    # )

    # print(output1.shape, output2.shape, output3.shape)
    # print("test preprocess func success\n\n")

    # #### test model #####

    # _, _, total_loss = model.meta_learner(
    #     [image_plus_stats1, image_plus_stats2, action_1, action_2]
    # )
    # loss_before, output, loss_after = model.meta_learner(input_tensor)
    # preupdate_loss = torch.sum(loss_before) / META_BATH_SIZE
    # per_update_loss = [
    #     torch.sum(loss_after(j)) / META_BATH_SIZE for j in range(NUM_UPDATES)
    # ]
    print("##########  total loss is ############\n {}".format(total_loss))


def main():

    # define index in the input data from state input to action input
    state_idx = datagenerator.state_idx
    img_idx = list(
        range(
            len(
                state_idx
            ),  # state_idx is [0,1,2...,dim_states] thus len(state_idx) = dim_states
            len(state_idx) + WHC,  # input is flatten image?
        )
    )

    processing and sample data
    datagenerator.generate_batches()
    all_files = datagenerator.make_batch_tensor()

    dataload = datasets(all_files, network_config)

    model = MODEL.daml_sim(state_idx=state_idx, img_idx=img_idx, config_params=network_config)

    train(dataload, model)


if __name__ == "__main__":
    main()
