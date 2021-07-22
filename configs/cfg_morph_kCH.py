import numpy as np
from os.path import join


class ConfigMorphV0:
    # data paths
    img_folder = '/hdd/2020/Research/datasets/Agedataset/img'
    # dataset_path = '/hdd2/2019/datasets/results_AADB/datasetImages_warp256'
    train_list = '/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_2_S1_train.txt'
    test_list = '/hdd/2020/Research/datasets/Agedataset/morph_fold/Setting_B/Setting_2_S2+S3_test.txt'
    age_minmax = [16, 77]
    age_range = np.log(age_minmax)  # in log scale
    dataset = 'morph_settingB'
    num_threads = 8

    # input config
    batch_size = 64
    width = 64
    height = 64
    n_comparator_output = 3

    # learning rate
    base_lr = 0.0001  # 0.0001
    lr_decay_rate = 0.5
    lr_decay_steps = 50000

    # training & test settings
    num_epoch = 1000
    test_freq = 1
    backbone_trainable = True
    hard_sample_ratio = 0

    # data statistics
    num_train = 7000
    num_test = 14000
    median_score = 0.5
    mean_score = 0.512344999

    # pairwise comparison
    # for voting method
    score_range = 1
    tau = 0.1
    N = 10
    delta = 0.2  # quantization step value
    K = 2
    do_augment = True
    age_feat_dim = 128
    identity_feat_dim = 1024 - 128
    feat_dim = 1024

    # DRC parameters
    num_quantization_level = 101
    age_levels = np.linspace(0, 1, num_quantization_level)
    clustering_by = 'kmeans'
    alpha = 0.1
