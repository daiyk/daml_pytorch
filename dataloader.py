import copy
import logging
import os
import glob
import tempfile
import pickle
from datetime import datetime
from collections import OrderedDict
from PIL import Image
import numpy as np
import random


# import tensorflow as tf
from utils import extract_demo_dict, Timer

# from tensorflow.python.platform import flags
from natsort import natsorted
from random import shuffle
from torch.utils.data import Dataset, DataLoader


class DataGenerator(object):
    def __init__(self, config, is_training=True):
        # Hyperparameters
        self.shuffle = False  # shuffle data for choosing val dataset
        self.config = config
        self.update_batch_size = config.get(
            "update_batch_size", 1
        )  # number of examples(demo video) used for inner gradient update (K for K-shot learning), number of configs for each task
        self.test_batch_size = 1  # number of examples(demo video) used for inner validation gradient update, #configs per task
        self.meta_batch_size = config.get(
            "meta_batch_size", 15
        )  # number of tasks sampled per meta-update
        self.frames = config.get(
            "frames", 100
        )  # time horizon of the demo videos, 50 for reach, 100 for push
        self.demo_gif_dir = config["demo_gif_dir"]
        self.gif_prefix = config.get(
            "gif_prefix", "object"
        )  # prefix of the video directory for each task
        self.restore_iter = config.get(
            "restored_iter", -1
        )  # iteration to load model, -1 for latest model
        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        demo_file = config.get(
            "demo_file"
        )  # path to the directory where demo files (tast files 不同的task，每个task有obj folder,里面有diff conditions) that containing robot states and actions are stored
        
        # need to be replaced with own demo file path 
        demo_file = natsorted(glob.glob(demo_file + "/*pkl"))
        self.dataset_size = len(demo_file)  # push sim has 768 tasks
        ## vision_reach: training_set_size = 750
        # if is_training and FLAGS.training_set_size != -1:
        #     tmp = demo_file[
        #         : FLAGS.training_set_size
        #     ]  # use the first #training_set_size as training data
        #     tmp.extend(
        #         demo_file[-FLAGS.val_set_size :]
        #     )  # use the last #val_set as testing data
        #     demo_file = tmp  # neglect [training_set:-val_set] data
        self.extract_supervised_data(demo_file, is_training=is_training)

        # we don't consider noisy conditions
        # if FLAGS.use_noisy_demos:
        #     self.noisy_demo_gif_dir = FLAGS.noisy_demo_gif_dir
        #     noisy_demo_file = FLAGS.noisy_demo_file
        #     self.extract_supervised_data(noisy_demo_file, noisy=True)

    def extract_supervised_data(self, demo_file, is_training=True, noisy=False):
        """
            Load the states and actions of the demos into memory.
            Args:
                demo_file: list of demo files where each file contains expert's states and actions of one task.
        """
        demos = extract_demo_dict(
            demo_file
        )  # demos = train_set_files(first train_set_size)+val_set_files(last val_set_size), subtract offset tasks
        # We don't need the whole dataset of simulated pushing.

        # demoX is the state, demoU is the actions (torques)
        # for each task in demos, it has shape [config_num, gif_frames, states/actions]
        # e.g. sim_push has 20 states and 7 torques
        # if FLAGS.experiment == "sim_push":
        # !!! sim_push give up first and last 6 configs per tasks
        for key in list(demos.keys()):
            demos[key]["demoX"] = demos[key]["demoX"][6:-6, :, :].copy()
            demos[key]["demoU"] = demos[key]["demoU"][6:-6, :, :].copy()
        n_folders = len(list(demos.keys()))  # number of train + val tasks

        # N_demos = sum all number of configs or number of demos (all demos with all tasks)
        N_demos = np.sum(demo["demoX"].shape[0] for i, demo in demos.items())
        self.state_idx = list(
            range(demos[0]["demoX"].shape[-1])
        )  # state_idx = [0,1...dim_states-1]
        self._dU = demos[0]["demoU"].shape[-1]  # _dU = dims of actions
        print("Number of demos: %d" % N_demos)
        idx = np.arange(n_folders)  # idx to every tasks [0,1,2,3,...,Num_tasks]
        if is_training:
            n_val = self.config.get("val_set_size", 76)  # number of demos for testing

            # shuffle data, not important
            if not hasattr(self, "train_idx"):
                if n_val != 0:
                    if not self.shuffle:
                        self.val_idx = idx[-n_val:]
                        self.train_idx = idx[:-n_val]
                    else:
                        self.val_idx = np.sort(
                            np.random.choice(idx, size=n_val, replace=False)
                        )
                        mask = np.array([(i in self.val_idx) for i in idx])
                        self.train_idx = np.sort(idx[~mask])
                else:
                    self.train_idx = idx
                    self.val_idx = []

            # Normalize the states if it's training.
            with Timer("Normalizing states"):
                # constructor: bias and scale are None
                if self.scale is None or self.bias is None:
                    # vstack all tasks' states [training_set, config, frames, states]
                    states = np.vstack(
                        (demos[i]["demoX"] for i in self.train_idx)
                    )  # hardcoded here to solve the memory issue

                    states = states.reshape(
                        -1, len(self.state_idx)
                    )  # [training_set*configs*frames, states]
                    # 1e-3 to avoid infs if some state dimensions don't change in the
                    # first batch of samples
                    # save bias and scale of all training set
                    self.scale = np.diag(1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                    self.bias = -np.mean(
                        states.dot(self.scale), axis=0
                    )  # calculate scaled states mean value and use as bias
                    # Save the scale and bias.
                    experiment = "sim_push"
                    with open("data/scale_and_bias_%s.pkl" % experiment, "wb") as f:
                        pickle.dump({"scale": self.scale, "bias": self.bias}, f)
                # [config, frames, states]->[config*frames, states]
                for key in list(demos.keys()):
                    demos[key]["demoX"] = demos[key]["demoX"].reshape(
                        -1, len(self.state_idx)
                    )
                    # ??? normalize states ???
                    demos[key]["demoX"] = (
                        demos[key]["demoX"].dot(self.scale) + self.bias
                    )
                    demos[key]["demoX"] = demos[key]["demoX"].reshape(
                        -1, self.frames, len(self.state_idx)
                    )  # [traning_set_num*configs(total configs), frames,states]
        # if not noisy:
        self.demos = demos
        # else:
        # self.noisy_demos = demos

    def generate_batches(self, noisy=False):
        with Timer("Generating batches for each iteration"):
            # if FLAGS.training_set_size != -1:
            #     # num of tasks - traning_set_sizes - test_set_sizes. These are the tasks that between training and testing and should be given up.
            #     offset = (
            #         self.dataset_size - FLAGS.training_set_size - FLAGS.val_set_size
            #     )
            # else:
            offset = 0
            img_folders = natsorted(
                glob.glob(self.demo_gif_dir + "/" + self.gif_prefix + "_*")
            )  # gif_prefix is 'object' to indicate the img_folder
            # print("img_folders: {}".format(img_folders))
            # print("train_idxs: {}".format(self.train_idx))
            train_img_folders = {
                i: img_folders[i] for i in self.train_idx
            }  # sample train folders according to the train idx
            val_img_folders = {
                i: img_folders[i + offset] for i in self.val_idx
            }  # we know that train_set + val_set = demos != total tasks only if offset != 0
            # if noisy:
            #     noisy_img_folders = natsorted(
            #         glob.glob(self.noisy_demo_gif_dir + self.gif_prefix + "_*")
            #     )
            #     noisy_train_img_folders = {
            #         i: noisy_img_folders[i] for i in self.train_idx
            #     }
            #     noisy_val_img_folders = {i: noisy_img_folders[i] for i in self.val_idx}
            TEST_PRINT_INTERVAL = 500  # with every this interval print the test result
            TOTAL_ITERS = self.config.get(
                "metatrain_iterations", 30000
            )  # number of metatraining iterations.
            self.all_training_filenames = []
            self.all_val_filenames = []
            self.training_batch_idx = {
                i: OrderedDict() for i in range(TOTAL_ITERS)
            }  # each batch represent each training iteration
            self.val_batch_idx = {
                i: OrderedDict()
                for i in TEST_PRINT_INTERVAL
                * np.arange(1, int(TOTAL_ITERS / TEST_PRINT_INTERVAL))
            }  # calculate validation iteration
            if noisy:
                self.noisy_training_batch_idx = {
                    i: OrderedDict() for i in range(TOTAL_ITERS)
                }
                self.noisy_val_batch_idx = {
                    i: OrderedDict()
                    for i in TEST_PRINT_INTERVAL
                    * np.arange(1, TOTAL_ITERS / TEST_PRINT_INTERVAL)
                }
            for itr in range(TOTAL_ITERS):
                sampled_train_idx = random.sample(
                    self.train_idx.tolist(), self.meta_batch_size
                )  # (num, sampled num) in all training tasks pool, sample #meta_train_idx tasks (number of tasks used for per iter training, k tasks learning) for training
                for idx in sampled_train_idx:  # each tasks sampled two demos
                    sampled_folder = train_img_folders[
                        idx
                    ]  # take the corresponding image folder for tasks folders of training meta_training_size
                    image_paths = natsorted(
                        os.listdir(sampled_folder)
                    )  # all demos in that tasks
                    # if FLAGS.experiment == "sim_push":
                    image_paths = image_paths[
                        6:-6
                    ]  # throw away first and last 6 gif/demos
                    try:
                        assert (
                            len(image_paths) == self.demos[idx]["demoX"].shape[0]
                        )  # ensure cut down both images and states/action
                    except AssertionError:
                        import pdb

                        pdb.set_trace()
                    # if noisy:
                    #     noisy_sampled_folder = noisy_train_img_folders[idx]
                    #     noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                    #     assert (
                    #         len(noisy_image_paths)
                    #         == self.noisy_demos[idx]["demoX"].shape[0]
                    #     )
                    # if not noisy:
                    # update_batch_size for inner gradient update, test_batch_size for inner validation and gradient update
                    sampled_image_idx = np.random.choice(
                        list(range(len(image_paths))),
                        size=self.update_batch_size
                        + self.test_batch_size,  # =2, for all demos of that for loop 'idx' task, sample two demos
                        replace=False,
                    )  # True
                    sampled_images = [
                        os.path.join(
                            sampled_folder, image_paths[i]
                        )  # join sampled_fold path and sampled_demo path
                        for i in sampled_image_idx
                    ]
                    # else:
                    #     noisy_sampled_image_idx = np.random.choice(
                    #         list(range(len(noisy_image_paths))),
                    #         size=self.update_batch_size,
                    #         replace=False,
                    #     )  # True
                    #     sampled_image_idx = np.random.choice(
                    #         list(range(len(image_paths))),
                    #         size=self.test_batch_size,
                    #         replace=False,
                    #     )  # True
                    #     sampled_images = [
                    #         os.path.join(noisy_sampled_folder, noisy_image_paths[i])
                    #         for i in noisy_sampled_image_idx
                    #     ]
                    #     sampled_images.extend(
                    #         [
                    #             os.path.join(sampled_folder, image_paths[i])
                    #             for i in sampled_image_idx
                    #         ]
                    #     )
                    self.all_training_filenames.extend(sampled_images)
                    self.training_batch_idx[itr][
                        idx
                    ] = sampled_image_idx  # len=2, each iteration, each idx(sampled tasks=15 for pushing), has values of two demos image folder idx (training+validate)
                    # if noisy:
                    #     self.noisy_training_batch_idx[itr][
                    #         idx
                    #     ] = noisy_sampled_image_idx

                # when test iter interval is reached,  do test. It is basically the same as the training stage. Sample update_batch_size imgs for one/multi gradient update
                # then sample test_batch_size for testing. Notice, no further update to the whole parameter set during testing.
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    sampled_val_idx = random.sample(
                        list(self.val_idx), self.meta_batch_size
                    )
                    for idx in sampled_val_idx:
                        sampled_folder = val_img_folders[idx]
                        image_paths = natsorted(os.listdir(sampled_folder))
                        # if FLAGS.experiment == "sim_push":
                        image_paths = image_paths[6:-6]
                        assert len(image_paths) == self.demos[idx]["demoX"].shape[0]
                        # if noisy:
                        #     noisy_sampled_folder = noisy_val_img_folders[idx]
                        #     noisy_image_paths = natsorted(
                        #         os.listdir(noisy_sampled_folder)
                        #     )
                        #     assert (
                        #         len(noisy_image_paths)
                        #         == self.noisy_demos[idx]["demoX"].shape[0]
                        #     )
                        # if not noisy:
                        sampled_image_idx = np.random.choice(
                            list(range(len(image_paths))),
                            size=self.update_batch_size + self.test_batch_size,
                            replace=False,
                        )  # True
                        sampled_images = [
                            os.path.join(sampled_folder, image_paths[i])
                            for i in sampled_image_idx
                        ]
                        # else:
                        #     noisy_sampled_image_idx = np.random.choice(
                        #         list(range(len(noisy_image_paths))),
                        #         size=self.update_batch_size,
                        #         replace=False,
                        #     )  # True
                        #     sampled_image_idx = np.random.choice(
                        #         list(range(len(image_paths))),
                        #         size=self.test_batch_size,
                        #         replace=False,
                        #     )  # True
                        #     sampled_images = [
                        #         os.path.join(noisy_sampled_folder, noisy_image_paths[i])
                        #         for i in noisy_sampled_image_idx
                        #     ]
                        #     sampled_images.extend(
                        #         [
                        #             os.path.join(sampled_folder, image_paths[i])
                        #             for i in sampled_image_idx
                        #         ]
                        #     )
                        self.all_val_filenames.extend(
                            sampled_images
                        )  # iters * per iters sampled image_paths
                        self.val_batch_idx[itr][idx] = sampled_image_idx
                        # if noisy:
                        #     self.noisy_val_batch_idx[itr][idx] = noisy_sampled_image_idx

    def make_batch_tensor(
        self, restore_iter=0, train=True
    ):  # used to generate input data per iteration by tf.train.batch(), should use it one time before start training for initial input data
        TEST_INTERVAL = 500
        batch_image_size = (
            self.update_batch_size + self.test_batch_size
        ) * self.meta_batch_size  # per task sampled configs * per iter sampled tasks=one iter required demos/configs=15*2
        if train:
            all_filenames = (
                self.all_training_filenames
            )  # all sampled demos and images path for iterations
            # if restore_iter > 0:
            #     all_filenames = all_filenames[
            #         batch_image_size * (restore_iter + 1) :
            #     ]  # calculate the restore point and corresponding size of image for rest iterations, it is arranged in the sequence of all iteration files
        else:
            all_filenames = self.all_val_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[
                    batch_image_size * (int(restore_iter / TEST_INTERVAL) + 1) :
                ]
        return all_filenames

        # im_height = network_config["image_height"]
        # im_width = network_config["image_width"]
        # num_channels = network_config["image_channels"]
        # # make queue for tensorflow to read from
        # filename_queue = tf.train.string_input_producer(
        #     tf.convert_to_tensor(all_filenames), shuffle=False
        # )
        # print("Generating image processing ops")
        # image_reader = tf.WholeFileReader()
        # _, image_file = image_reader.read(filename_queue)
        # image = tf.image.decode_gif(image_file)
        # # should be T x C x W x H
        # image.set_shape((self.frames, im_height, im_width, num_channels))
        # image = tf.cast(image, tf.float32)
        # image /= 255.0
        # # if FLAGS.hsv:  # convert to HSV format
        # #     eps_min, eps_max = 0.5, 1.5
        # #     assert eps_max >= eps_min >= 0
        # #     # convert to HSV only fine if input images in [0, 1]
        # #     img_hsv = tf.image.rgb_to_hsv(image)s
        # #     img_h = img_hsv[..., 0]  # hue
        # #     img_s = img_hsv[..., 1]  # saturation
        # #     img_v = img_hsv[..., 2]  # value
        # #     eps = tf.random_uniform([self.T, 1, 1], eps_min, eps_max)
        # #     img_v = tf.clip_by_value(eps * img_v, 0.0, 1.0)
        # #     img_hsv = tf.stack([img_h, img_s, img_v], 3)
        # #     image_rgb = tf.image.hsv_to_rgb(img_hsv)
        # #     image = image_rgb
        # image = tf.transpose(
        #     image, perm=[0, 3, 2, 1]
        # )  # transpose to mujoco setting for images [T,H,W,C]->[T,C,W,H]
        # image = tf.reshape(image, [self.frames, -1])
        # num_preprocess_threads = 1  # TODO - enable this to be set to >1
        # min_queue_examples = 64  # 128 #256
        # print("Batching images")
        # images = tf.train.batch(
        #     [image],
        #     batch_size=batch_image_size,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3 * batch_image_size,
        # )  # here images is the total image files that is divided to each #batch_image_size for each meta_training iteration
        # #!!! tf.train.batch automatically load the next batch after each call of sess.run() touch the images
        # all_images = []
        # for i in range(
        #     self.meta_batch_size
        # ):  # fetch images of all tasks for this iteration
        #     image = images[
        #         i
        #         * (self.update_batch_size + self.test_batch_size) : (i + 1)
        #         * (self.update_batch_size + self.test_batch_size)
        #     ]
        #     image = tf.reshape(
        #         image,
        #         [(self.update_batch_size + self.test_batch_size) * self.frames, -1],
        #     )
        #     all_images.append(image)
        # return tf.stack(all_images)

    def generate_data_batch(self, itr, train=True):
        # used to generate ground truth
        if train:
            demos = {
                key: self.demos[key].copy() for key in self.train_idx
            }  # self.demos is the total data files paths, train_idx has all training tasks pool
            idxes = self.training_batch_idx[
                itr
            ]  # training_batch_idx = [iters][idxes] image folder idx for each this "iter" at sampled tasks, idxes is dictionary, ['sampled task idex': 'image folders idexes']
            # if FLAGS.use_noisy_demos:
            #     noisy_demos = {
            #         key: self.noisy_demos[key].copy() for key in self.train_idx
            #     }
            #     noisy_idxes = self.noisy_training_batch_idx[itr]
        else:
            demos = {key: self.demos[key].copy() for key in self.val_idx}
            idxes = self.val_batch_idx[itr]
            # if FLAGS.use_noisy_demos:
            #     noisy_demos = {
            #         key: self.noisy_demos[key].copy() for key in self.val_idx
            #     }
            #     noisy_idxes = self.noisy_val_batch_idx[itr]
        batch_size = self.meta_batch_size  # batch number of tasks=15 per iteration
        update_batch_size = (
            self.update_batch_size
        )  # number of configs per tasks for inner training =1
        test_batch_size = (
            self.test_batch_size
        )  # number of configs per tasks for inner validation=1
        # if not FLAGS.use_noisy_demos:
        U = [
            demos[k]["demoU"][v].reshape(
                (test_batch_size + update_batch_size) * self.frames, -1
            )
            for k, v in list(
                idxes.items()
            )  # k is the sampled task folder idx, v is the sampled image folder idx/ groudntruth index.  are the idx of [meta_batch_size, pre_plus_post_update_demos * frames, actions]
        ]  # demos: all tasks(folders) , k, v identify task folders and image demos path for this iteration
        U = np.array(U)
        X = [
            demos[k]["demoX"][v].reshape(
                (test_batch_size + update_batch_size) * self.frames, -1
            )
            for k, v in list(
                idxes.items()
            )  # [meta_batch_size, pre_plus_post_update_demos * frames, actions]
        ]
        X = np.array(X)
        # else:
        #     noisy_U = [
        #         noisy_demos[k]["demoU"][v].reshape(update_batch_size * self.frames, -1)
        #         for k, v in list(noisy_idxes.items())
        #     ]
        #     noisy_X = [
        #         noisy_demos[k]["demoX"][v].reshape(update_batch_size * self.frames, -1)
        #         for k, v in list(noisy_idxes.items())
        #     ]
        #     U = [
        #         demos[k]["demoU"][v].reshape(test_batch_size * self.frames, -1)
        #         for k, v in list(idxes.items())
        #     ]
        #     U = np.concatenate((np.array(noisy_U), np.array(U)), axis=1)
        #     X = [
        #         demos[k]["demoX"][v].reshape(test_batch_size * self.frames, -1)
        #         for k, v in list(idxes.items())
        #     ]
        #     X = np.concatenate((np.array(noisy_X), np.array(X)), axis=1)
        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.state_idx)  # states_idx=[0,1,2,.....,dim_states]
        return X, U  # get groundtruth for this iteration


class datasets(Dataset):
    def __init__(self, all_filenames, config, training=True):
        self.train = training
        self.all_files = all_filenames  # iters*batch*(train_size+update_size)
        self.config = config
        self.im_height = config["image_height"]
        self.im_width = config["image_width"]
        self.num_channels = config["image_channels"]

    def __getitem__(self, idx):
        # getitem should return 2 idxes per batch
        batchsize = self.config.get("update_batch_size", 1) + self.config.get(
            "test_batch_size", 1
        )
        images_path = self.all_files[idx * batchsize : (idx + 1) * batchsize]
        images = []
        # read images from images_path
        for i in range(len(images_path)):
            imgs = []
            im = Image.open(images_path[i])
            imgs.append(im)
            try:
                while True:
                    im.seek(im.tell() + 1)
                    imgs.append(im)
            except EOFError:
                pass
            imgs_np = [np.asarray(imgs[i].convert("RGB")) for i in range(len(imgs))]

            # reshape the image
            imgs_np = [img / 255.0 for img in imgs_np]
            imgs_np = np.stack(imgs_np, axis=0)  # [T,H,W,C]
            imgs_np = np.reshape(
                np.transpose(imgs_np, [0, 3, 2, 1]),
                [self.config.get("frames", 100), -1],
            )  # reshape to [T,C*W*H]

            images.append(imgs_np)
        return np.reshape(
            np.stack(images, axis=0), [self.config.get("frames", 100) * batchsize, -1]
        )

    def __len__(self):
        batchsize = self.config.get("update_batch_size", 1) + self.config.get(
            "test_batch_size", 1
        )
        return int(len(self.all_files) / batchsize)  # length = iterations * batch_num

