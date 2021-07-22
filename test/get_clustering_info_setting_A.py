import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score as compute_nmi

sys.path.append('..')
from configs.cfg_morph_clustering_kCH_setting_A import ConfigMorphV0 as Config
from networks import model_ae
from utils.clustering_utils_v6_alpha_repulsive import assign_memberships_by_nn_rule
from utils.utils import save_or_load_feature
from utils.utils import write_log


def main():
    def green(x):
        return '\033[92m' + x + '\033[0m'

    def blue(x):
        return '\033[94m' + x + '\033[0m'

    # --- select GPU to use
    GPU = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # specify which GPU(s) to be used
    print(f'USE GPU {GPU}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- load configs
    config = Config()
    log_path = '../../results/EXPERIMENT_FOLDER'   ### <--- EDIT HERE
    to_load_epoch = 120

    # --- generate log files
    ckpt_file = f'{log_path}/checkpoints/ckpt_ep{to_load_epoch}'
    centroid_file = f'{log_path}/centroids/centroids_ep{to_load_epoch}'
    summary_path = f'{log_path}/summary'
    analysis_path = f'{log_path}/analysis'
    log_file = open(f'{log_path}/clustering_info.txt', 'w')

    # --- make network model
    model = model_ae.spherical_v0(config)
    model.summary()

    try:
        model.load_weights(ckpt_file)
        write_log(log_file, f'Parameters are loaded from {ckpt_file}')
    except:
        write_log(log_file, f'Network are initialized with random variables')

    train_data = pd.read_csv(config.train_list, sep=",")
    test_data = pd.read_csv(config.test_list, sep=",")

    train_filelist = np.array([f'{config.img_folder}/{train_data["database"][idx]}/{train_data["filename"][idx]}' for idx in range(len(train_data))])
    train_labels = np.array(train_data['age'])
    train_gender = np.array(train_data['gender'])
    train_race = np.array(train_data['race'])

    test_filelist = np.array([f'{config.img_folder}/{test_data["database"][idx]}/{test_data["filename"][idx]}' for idx in range(len(test_data))])
    test_labels = np.array(test_data['age'])
    test_race = np.array(test_data['race'])
    test_gender = np.array(test_data['gender'])

    print(f'{config.dataset} dataset is successfully loaded!')

    # extract encoder part
    feature_extractor = model.get_layer('encoder')
    encoder = keras.Model(inputs=feature_extractor.input, outputs=feature_extractor.output)

    train_features = save_or_load_feature(f'{log_path}/features_ckpt_{to_load_epoch}', train_filelist, config.feat_dim, encoder, config)
    train_attr_features = normalize(train_features[:, config.age_feat_dim:config.age_feat_dim+config.chain_feat_dim])
    centroids = pkl.load(open(centroid_file, "rb"))
    train_memberships = assign_memberships_by_nn_rule(train_attr_features, centroids)

    # extract test features
    test_features = save_or_load_feature(f'{log_path}/test_features_ckpt_{to_load_epoch}', test_filelist, config.feat_dim, encoder, config)
    test_attr_features = normalize(test_features[:, config.age_feat_dim:config.age_feat_dim+config.chain_feat_dim])
    test_memberships = assign_memberships_by_nn_rule(test_attr_features, centroids)

    print('TRAIN')
    for k in range(config.K):
        print(f'cluster {k} :  {len(np.argwhere(train_memberships == k).flatten())}')
    print('TEST')
    for k in range(config.K):
        print(f'cluster {k} :  {len(np.argwhere(test_memberships == k).flatten())}')

    print(f'train_nmi: {compute_nmi(train_gender, train_memberships)}')
    print(f'test_nmi {compute_nmi(test_gender, test_memberships)}')

    sims = np.matmul(train_attr_features, np.transpose(train_attr_features))
    sims_test = np.matmul(test_attr_features, np.transpose(train_attr_features))

    np.save(f'{log_path}/tr_memberships', train_memberships)
    np.save(f'{log_path}/te_memberships', test_memberships)
    np.save(f'{log_path}/sims_tr_tr.npy', sims)
    np.save(f'{log_path}/sims_te_tr.npy', sims_test)


if __name__ == "__main__":
    main()