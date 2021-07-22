from datetime import datetime
import os
import sys

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import normalize
import tensorflow as tf
import tensorflow.keras as keras


sys.path.append('..')
from configs.cfg_morph_estimation_kCH_setting_A import ConfigMorphV0 as Config
from networks import model_comparator
from utils.comparison_utils import compute_ternary_order_fixed_ref
from utils.comparison_utils import compute_ternary_order_fixed_base
from utils.comparison_utils import one_step_voting_ternary_log, soft_voting_ternary_log
from utils.utils import save_or_load_feature_v2
from utils.utils import load_images
from utils.utils import write_log


def clustering_imgs_save(memberships, filelists, analysis_path):
    K = max(memberships) + 1
    if not os.path.exists(f'{analysis_path}/K_{K}'): os.mkdir(f'{analysis_path}/K_{K}')
    for k in range(K):
        if not os.path.exists(f'{analysis_path}/K_{K}/{k}'): os.mkdir(f'{analysis_path}/K_{K}/{k}')

    for idx, (i_cluster, filename) in enumerate(zip(memberships, filelists)):
        if idx % 500 == 0:
            print(f'{idx} saved / {memberships.shape[0]}')
        os.system(f'cp {filename} {analysis_path}/K_{K}/{i_cluster}')


def main():
    # --- select GPU to use
    GPU = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # specify which GPU(s) to be used
    print(f'USE GPU {GPU}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print('gpu', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
        print('memory growth:', tf.config.experimental.get_memory_growth(gpu))

    # --- load configs
    config = Config()
    experiment_name = 'vgg16mdba_t0.1_2CH_Setting_A_20200915-210722'  ### <--- EDIT HERE 
    ckpt_name = 'ckpt_0.8740808963775635'  ### <--- EDIT HERE 
    ref_file = 'ref_by_attr.npy'

    # --- generate log files
    log_path = f'../results/results_morph/{experiment_name}'
    ckpt_path = f'{log_path}/checkpoints'
    ckpt_to_load = f'{ckpt_path}/{ckpt_name}'

    batch_size_for_comp = 1024
    # ref_filelist = ['ref_list_ep170_by_acc_N3_L66_T3_cluster_id0.npy',
    #                 'ref_list_ep170_by_acc_N3_L66_T3_cluster_id1.npy',
    #                 ]
    batch_size_for_comp = 66 * 3

    # --- create or open log file
    log_file_path = f'{log_path}/test_log_{ckpt_name}.txt'
    if os.path.exists(log_file_path):
        log_file = open(log_file_path, 'a')
    else:
        log_file = open(log_file_path, 'w')

    # --- record the config of experiment
    config_dict = vars(Config)
    write_log(log_file, '*' * 100)
    for key in config_dict.keys():
        if not key.startswith('_'):
            write_log(log_file, f'{key} : {config_dict[key]}')
    write_log(log_file, '*' * 100 + '\n')
    #
    # for ref_file in ref_filelist:
    #     write_log(log_file, ref_file)

    # --- load data
    train_data = pd.read_csv(config.train_list, sep=",")
    test_data = pd.read_csv(config.test_list, sep=",")

    train_filelist = np.array(
        [f'{config.img_folder}/{train_data["database"][idx]}/{train_data["filename"][idx]}' for idx in
         range(len(train_data))])
    train_labels = np.array(train_data['age'])
    train_genders = np.array(train_data['gender'])
    train_race = np.array(train_data['race'])

    test_filelist = np.array(
        [f'{config.img_folder}/{test_data["database"][idx]}/{test_data["filename"][idx]}' for idx in
         range(len(test_data))])
    test_labels = np.array(test_data['age'])
    test_race = np.array(test_data['race'])

    # --- load clustering info
    print(f'{config.dataset} dataset is successfully loaded!')

    # --- make network model
    model = model_comparator.vgg_mdba(config)
    model.summary()
    write_log(log_file, f'Model name: {model.name}')
    try:
        model.load_weights(ckpt_to_load)
        write_log(log_file, f'Parameters are loaded from {ckpt_to_load}')
    except:
        write_log(log_file, f'Network are initialized with IMAGENET feature')

    # extract encoder part
    feature_extractor = model.get_layer('feature_extractor')
    feature_extractor = keras.Model(inputs=feature_extractor.input, outputs=feature_extractor.output)

    comp_input = keras.Input(512 * 2)
    l1 = model.get_layer('dense_1')
    l2 = model.get_layer('batch_normalization_1')
    l3 = model.get_layer('activation_1')
    l4 = model.get_layer('dense_2')
    l5 = model.get_layer('batch_normalization_2')
    l6 = model.get_layer('activation_2')
    l7 = model.get_layer('dense_3')

    x = l1(comp_input)
    x = l2(x)
    x = l3(x)
    x = l4(x)
    x = l5(x)
    x = l6(x)
    output = l7(x)

    comparator = keras.Model(comp_input, output, name='comparator')

    train_features = save_or_load_feature_v2(f'{log_path}/train_features_{ckpt_name}', train_filelist,512, feature_extractor, config)
    test_features = save_or_load_feature_v2(f'{log_path}/test_features_{ckpt_name}', test_filelist,512, feature_extractor, config)

    #
    total_preds = []
    total_soft_preds = []
    total_gt = []
    total_comparison_acc = []
    ref_idxs_list = np.load(f'{log_path}/{ref_file}', allow_pickle=True)

    # --- infer age
    for base_idx in range(config.num_test):
        ref_idxs = ref_idxs_list[base_idx]
        ref_features = train_features[ref_idxs]
        ref_labels = train_labels[ref_idxs]
        num_ref = len(ref_labels)
        batch_base = np.zeros((batch_size_for_comp, 512), dtype=np.float32)
        batch_base[:, ...] = test_features[base_idx]
        order_list = []
        age_list = []
        prob_list = []

        for batch_idx in range(np.ceil(num_ref / batch_size_for_comp).astype(np.int32)):
            start_idx = batch_size_for_comp * batch_idx
            end_idx = min(start_idx + batch_size_for_comp, num_ref)
            batch_ref = ref_features[start_idx:end_idx, ...]
            batch_label = ref_labels[start_idx:end_idx]

            # 1. infer the ordering relation
            if end_idx - start_idx < batch_size_for_comp:
                batch_base = np.zeros_like(batch_ref)
                batch_base[:, ...] = test_features[base_idx]

            batch_pair = tf.concat((batch_base, batch_ref), axis=-1)
            preds = comparator(batch_pair, training=False)
            preds = tf.nn.softmax(preds, axis=-1)
            order_pred = np.argmax(preds, axis=-1)
            prob_list.append(preds.numpy())
            order_list.append(order_pred)
            age_list.append(batch_label)
        order_list = np.concatenate(order_list, axis=0)
        age_list = np.concatenate(age_list, axis=0)
        prob_list = np.concatenate(prob_list, axis=0)

        # 2. hard voting
        gt_order_list = compute_ternary_order_fixed_base(np.log(test_labels[base_idx]), np.log(age_list), config.tau)
        comparison_acc = np.sum(gt_order_list == order_list) / num_ref

        pred_score, voting_result = one_step_voting_ternary_log(order_list, np.log(age_list), config.tau,
                                                                config.age_levels)

        # 3. soft voting
        soft_pred, _, _ = soft_voting_ternary_log(prob_list, np.log(age_list), config.tau, config.age_levels)

        total_preds.append(pred_score)
        total_soft_preds.append(soft_pred)
        total_gt.append(test_labels[base_idx])
        total_comparison_acc.append(comparison_acc)
        print(f'infer the score: {base_idx} / {config.num_test}')

    total_preds = np.array(total_preds)
    total_soft_preds = np.array(total_soft_preds)
    total_gt = np.array(total_gt)
    total_comparison_acc = np.array(total_comparison_acc)
    total_MAE = np.abs(total_preds - total_gt)
    total_MAE_soft = np.abs(total_soft_preds - total_gt)

    # --- measure MAE
    MAE_metric = keras.metrics.MeanAbsoluteError()
    MAE_metric(total_gt, total_preds)

    # --- measure CS (MAE <= 5)
    n_correct_CS = np.sum(total_MAE <= 5)
    CS = n_correct_CS / len(test_labels)

    n_correct_CS_soft = np.sum(total_MAE_soft <= 5)
    CS_soft = n_correct_CS_soft / len(test_labels)

    write_log(log_file, '\n+ ------------------------------------------------------------ +')
    write_log(log_file, '|                           TEST                               |')  # MAE: 4.23  CS: 73.2%
    write_log(log_file, '| ============================================================ |')
    write_log(log_file, '|        MAE        |      CS (%)      |   Comparison Acc. (%) |')
    write_log(log_file, '+ ------------------------------------------------------------ +')
    write_log(log_file, f'|      {MAE_metric.result():.3f}        |      {CS * 100:.3f}      |         {np.mean(total_comparison_acc) * 100:.3f}        |')
    write_log(log_file, f'|      {np.mean(total_MAE_soft):.3f}        |      {CS_soft * 100:.3f}      |         {np.mean(total_comparison_acc) * 100:.3f}        |')

    write_log(log_file, '+ ------------------------------------------------------------ +')




if __name__ == "__main__":
    main()