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


def main():
    # --- select GPU to use
    GPU = '0'
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

    # --- load data
    train_data = pd.read_csv(config.train_list, sep=",")
    train_labels = np.array(train_data['age'])

    clustering_info_folder = '/hdd/2020/Research/Clustering/clustering_v5/results/results_morph/EXPERIMENT_FOLDER'   ### <-- EDIT HERE
    # --- load clustering info
    tr_chain_file = f'{clustering_info_folder}tr_memberships.npy'
    te_chain_file = f'{clustering_info_folder}te_memberships.npy'
    tr_sims_file = f'{clustering_info_folder}sims_tr_tr.npy'
    te_sims_file = f'{clustering_info_folder}sims_te_tr.npy'

    tr_memberships = np.load(tr_chain_file)
    te_memberships = np.load(te_chain_file)
    tr_sims = np.load(tr_sims_file)
    te_sims = np.load(te_sims_file)

    print(f'{config.dataset} dataset is successfully loaded!')

    total_ref = []
    cluster_dict = dict()
    for k in range(config.K):
        cluster_dict[k] = np.argwhere(tr_memberships==k).flatten()

    for idx in range(len(te_memberships)):
        cur_k = te_memberships[idx]
        k_idxs = cluster_dict[cur_k]
        k_labels = train_labels[k_idxs]
        if idx% 200 == 0:
            print(f'{idx} / {len(te_memberships)}')
        ref_list = []
        for age in range(config.age_minmax[0], config.age_minmax[1]+1):
            cur_age_idx = k_idxs[np.argwhere(k_labels==age).flatten()]
            sim_idxs = np.argsort(-te_sims[idx, cur_age_idx])[0:config.N]
            ref_list.append(cur_age_idx[sim_idxs])
        total_ref.append(np.concatenate(ref_list))

    np.save(f'{clustering_info_folder}ref_by_attr', total_ref)


if __name__ == "__main__":
    main()