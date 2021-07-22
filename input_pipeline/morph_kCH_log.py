import numpy as np
import pandas as pd
import tensorflow as tf

from utils.utils import random_brightness
from utils.utils import random_contrast
from utils.utils import random_crop_image
from utils.utils import random_flip_image
from utils.utils import random_hue
from utils.utils import random_resize_image
from utils.utils import random_saturation
from utils.utils import load_one_image


def get_tf_mapping(config):
    """ augment data by randomly [flip, resize, crop], [adjust brightness, saturation, hue, contrast]
    """
    def tf_map(base_img, ref_img, order_label, chain_label):
        img = tf.concat([base_img, ref_img], axis=0)
        img = random_flip_image(img)     # random flip
        # img = random_brightness(img)     # max_delta = 32/255
        # img = random_saturation(img)     # 0.5 to 1.5
        # img = random_hue(img)            # max_delta = 0.2
        # img = random_contrast(img)       # 0.5 to 1.5
        img = random_resize_image(img)   # rescale within [1.05, 1.25]
        img = random_crop_image(img, [config.batch_size*2, config.width, config.height, 3])
        base_img, ref_img = tf.split(img, 2, axis=0)
        return base_img, ref_img, order_label, chain_label

    return tf_map


def get_indices_in_range(search_range, ages):
    return np.argwhere(np.logical_and(search_range[0] <= ages, ages <= search_range[1]))


def find_reference(base_age, ages, tau, hard_sample_ratio=0, hard_level=1.5, min_age=15, max_age=80, epsilon=1e-5):
    order = np.random.randint(0, 3)
    is_hard_sample = np.random.choice(2, 1, p=[1-hard_sample_ratio, hard_sample_ratio])[0]
    ref_idx = -1
    debug_flag = 0
    while ref_idx == -1:
        if debug_flag == 3:
            raise ValueError(f'Failed to find reference... base_score: {base_age}')
        if order == 0:    # base_score > ref_score + tau
            ref_range_min = max(base_age-(tau*hard_level), min_age) if is_hard_sample else min_age
            ref_range_max = base_age - tau - epsilon
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ages)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order+1) % 3
                debug_flag += 1
                pass

        elif order == 1:  # base_score ~= ref_score
            ref_range_min = base_age - tau
            ref_range_max = base_age + tau
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ages)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order + 1) % 3
                debug_flag += 1
                pass

        else:             # base_score < ref_score
            ref_range_min = base_age + tau + epsilon
            ref_range_max = min(base_age+(tau*hard_level)+epsilon, max_age) if is_hard_sample else max_age
            candidates = get_indices_in_range([ref_range_min, ref_range_max], ages)
            if len(candidates) > 0:
                ref_idx = candidates[np.random.choice(len(candidates), 1)[0]][0]
            else:
                order = (order + 1) % 3
                debug_flag += 1
                pass

    return order, ref_idx


def get_batch_gen_train(train_data, train_memberships, cluster_dict, config):
    """
    A function defining the batch generator for each split.
    It should return the generator, generated data types, and generated shapes

    :param images:
    :param labels:
    :param split:
    :param setting:
    :return: gen_func, gen_types, gen_shapes
    """

    def batch_generator():
        base_img_list = []
        ref_img_list = []
        order_list = []
        cluster_list = []
        num_in_batch = 0

        # randomly shuffle indices
        gen_indices = np.random.permutation(train_data.shape[0])
        ages_log = np.log(np.array(train_data['age']))

        # --- generator loop
        for i_data in gen_indices:
            base_img_path = f'{config.img_folder}/{train_data["database"][i_data]}/{train_data["filename"][i_data]}'
            base_img = load_one_image(base_img_path, config.width, config.height)
            base_age = ages_log[i_data]
            base_membership = train_memberships[i_data]
            ref_cands_idxs = cluster_dict[base_membership]
            ref_cands_ages = ages_log[ref_cands_idxs]

            order, ref_sel_idx = find_reference(base_age, ref_cands_ages, tau=config.tau, hard_sample_ratio=config.hard_sample_ratio,
                                                  min_age=config.age_range[0],
                                                  max_age=config.age_range[1])

            ref_idx = ref_cands_idxs[ref_sel_idx]
            ref_img_path = f'{config.img_folder}/{train_data["database"][ref_idx]}/{train_data["filename"][ref_idx]}'
            ref_img = load_one_image(ref_img_path, config.width, config.height)

            # add data to batch
            base_img_list.append(base_img)
            ref_img_list.append(ref_img)
            order_list.append(order)
            cluster_list.append(base_membership)

            # update num elements in batch
            num_in_batch += 1

            if num_in_batch == config.batch_size:
                yield (np.concatenate(np.expand_dims(base_img_list, axis=0), axis=0),
                       np.concatenate(np.expand_dims(ref_img_list, axis=0), axis=0),
                       np.array(order_list, dtype=np.int32),
                       np.array(cluster_list, dtype=np.int32)
                       )
                base_img_list = []
                ref_img_list = []
                order_list = []
                cluster_list = []
                num_in_batch = 0

    gen_dtypes = (tf.float32, tf.float32, tf.int32, tf.int32)
    gen_shapes = ([config.batch_size, config.width, config.height, 3],
                  [config.batch_size, config.width, config.height, 3],
                  [config.batch_size],
                  [config.batch_size])

    return batch_generator, gen_dtypes, gen_shapes


def get_batch_gen_test(test_data, train_data, test_memberships, cluster_dict, config):
    """
    A function defining the batch generator for each split.
    It should return the generator, generated data types, and generated shapes

    :param images:
    :param labels:
    :param split:
    :param setting:
    :return: gen_func, gen_types, gen_shapes
    """

    def batch_generator():
        base_img_list = []
        ref_img_list = []
        order_list = []
        cluster_list = []
        num_in_batch = 0

        # gen indices
        gen_indices = np.arange(test_data.shape[0])
        base_ages_log = np.log(np.array(test_data['age']))
        ref_ages_log = np.log(np.array(train_data['age']))

        # --- generator loop
        for i_data in gen_indices:
            base_img_path = f'{config.img_folder}/{test_data["database"][i_data]}/{test_data["filename"][i_data]}'
            base_img = load_one_image(base_img_path, config.width, config.height)
            base_age = base_ages_log[i_data]
            base_membership = test_memberships[i_data]
            ref_cands_idxs = cluster_dict[base_membership]
            ref_cands_ages = ref_ages_log[ref_cands_idxs]

            order, ref_sel_idx = find_reference(base_age, ref_cands_ages, tau=config.tau,
                                             min_age=config.age_range[0],
                                             max_age=config.age_range[1])
            ref_idx = ref_cands_idxs[ref_sel_idx]

            ref_img_path = f'{config.img_folder}/{train_data["database"][ref_idx]}/{train_data["filename"][ref_idx]}'
            ref_img = load_one_image(ref_img_path, config.width, config.height)

            # add data to batch
            base_img_list.append(base_img)
            ref_img_list.append(ref_img)
            order_list.append(order)
            cluster_list.append(base_membership)

            # update num elements in batch
            num_in_batch += 1

            if num_in_batch == config.batch_size:
                yield (np.concatenate(np.expand_dims(base_img_list, axis=0), axis=0),
                       np.concatenate(np.expand_dims(ref_img_list, axis=0), axis=0),
                       np.array(order_list, dtype=np.int32),
                       np.array(cluster_list, dtype=np.int32))
                base_img_list = []
                ref_img_list = []
                order_list = []
                cluster_list = []
                num_in_batch = 0

    gen_dtypes = (tf.float32, tf.float32, tf.int32, tf.int32)
    gen_shapes = ([config.batch_size, config.width, config.height, 3],
                  [config.batch_size, config.width, config.height, 3],
                  [config.batch_size],
                  [config.batch_size])

    return batch_generator, gen_dtypes, gen_shapes


def input_pipeline(config, train_memberships=None, test_memberships=None, chain_dict=None):
    """ gets input pipeline """

    # --- load data
    train_data = pd.read_csv(config.train_list, sep=",")
    test_data = pd.read_csv(config.test_list, sep=",")

    # chain
    if config.chain_by == 'gender':
        train_memberships = np.array(train_data['gender'], dtype=np.int32)
        test_memberships = np.array(test_data['gender'], dtype=np.int32)
        chain_dict = dict()
        for k in range(config.K):
            chain_dict[k] = np.argwhere(train_memberships==k).flatten()
    elif config.chain_by == 'database':
        db2cluster = dict()
        db2cluster['morph'] = 0
        db2cluster['afad'] = 1
        db2cluster['utk'] = 1

        train_database = np.array(train_data['database'])
        test_database = np.array(test_data['database'])
        train_memberships = np.array([db2cluster[db] for db in train_database])
        test_memberships = np.array([db2cluster[db] for db in test_database])

        chain_dict = dict()
        for k in range(config.K):
            chain_dict[k] = np.argwhere(train_memberships == k).flatten()

    elif config.clustering_by == 'kmeans':
        print('clustering by kmeans')
    else:
        raise ValueError(f'Check the clustering rule {config.clustering_by}')

    # gen function
    gen_function, gen_dtypes, gen_shapes = get_batch_gen_train(train_data, train_memberships, chain_dict, config)
    gen_function_test, _, _ = get_batch_gen_test(test_data, train_data, test_memberships, chain_dict, config)

    # tf dataset
    train_dataset = tf.data.Dataset.from_generator(gen_function, gen_dtypes, gen_shapes)
    test_dataset = tf.data.Dataset.from_generator(gen_function_test, gen_dtypes, gen_shapes)

    # map function
    if config.do_augment:
        map_func = get_tf_mapping(config)
        train_dataset = train_dataset.map(map_func=map_func, num_parallel_calls=config.num_threads)

    # prefetch data
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == "__main__":
    import sys
    import os
    sys.path.append('../..')
    from configs.cfg_balanced_kCH_clustering import ConfigBalancedV0 as Config
    from utils.visualize_util import show_img
    import cv2

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used
    config = Config()
    config.clustering_by = 'kmeans'
    config.K = 1

    train_memberships = np.zeros([config.num_train, ], dtype=np.int32)
    cluster_dict = dict()
    cluster_dict[0] = np.arange(config.num_train)
    test_memberships = np.zeros([config.num_test, ], dtype=np.int32)
    centroids = None
    tr, te = input_pipeline(config, train_memberships, test_memberships, cluster_dict)

    it = iter(te)
    base_imgs, ref_imgs, orders, chains, neg_chains = it.next()
    cv2.imshow("Display window", base_imgs[0].numpy())
    cv2.imshow("Display window", ref_imgs[0].numpy())
    show_img(base_imgs[0].numpy())
    show_img(ref_imgs[0].numpy())

    print('done.')
