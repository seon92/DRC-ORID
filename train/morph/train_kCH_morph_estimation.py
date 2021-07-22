from datetime import datetime
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('..')
from input_pipeline.morph_kCH_log import input_pipeline
from configs.cfg_morph_estimation_kCH_setting_A import ConfigMorphV0 as Config
from networks import model_comparator
from utils.loss_util import compute_cc_loss
from utils.utils import write_log


@tf.function
def train_one_step(model, optimizer, data, labels):
    with tf.GradientTape() as tape:
        preds = model(data)
        one_hot_label = tf.one_hot(labels, 3)
        loss = compute_cc_loss(one_hot_label, preds, label_smoothing=0.2*np.random.rand(1)[0])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return preds, loss


@tf.function
def test_one_step(model, data, labels):
    preds = model(data, training=False)
    one_hot_label = tf.one_hot(labels, 3)
    loss = compute_cc_loss(one_hot_label, preds)
    return preds, loss


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
    experiment_name = f'vgg16mdba_t{config.tau}_{config.K}CH_{config.dataset}' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- generate log files
    log_path = f'../../results/results_morph/{experiment_name}'

    tr_cluster_file = 'EXPERIMENT_FOLDER/tr_memberships.npy'   ### EDIT HERE !
    te_cluster_file = 'EXPERIMENT_FOLDER/te_memberships.npy'
    tr_sims_file = 'EXPERIMENT_FOLDER/sims_tr_tr.npy'
    te_sims_file = 'EXPERIMENT_FOLDER/sims_te_tr.npy'

    ckpt_file = 'EXPERIMENT_FOLDER/CKPT_NAME'   ### EDIT HERE if you want to resume training
    # ckpt_file = '/hdd2/2020/Research/Clustering/clustering_v5/results/results_morph/comparator_t0.1_2CH_vgg16small_20200724-062929/checkpoints/ckpt_0.8291284441947937'

    ckpt_path = f'{log_path}/checkpoints'
    summary_path = f'{log_path}/summary'
    if not os.path.exists(log_path): os.mkdir(log_path)
    if not os.path.exists(ckpt_path): os.mkdir(ckpt_path)
    if not os.path.exists(summary_path): os.mkdir(summary_path)
    log_file = open(f'{log_path}/log.txt', 'w')

    # --- record the config of experiment
    config_dict = vars(Config)
    write_log(log_file, '*' * 100)
    for key in config_dict.keys():
        if not key.startswith('_'):
            write_log(log_file, f'{key} : {config_dict[key]}')
    write_log(log_file, '*' * 100 + '\n')

    # --- load clustering info
    tr_memberships = np.load(tr_cluster_file)
    te_memberships = np.load(te_cluster_file)
    tr_sims = np.load(tr_sims_file)
    te_sims = np.load(te_sims_file)

    cluster_dict = dict()
    for k in range(config.K):
        cluster_dict[k] = np.argwhere(tr_memberships==k).flatten()


    # --- build data pipeline
    train_data, test_data = input_pipeline(config, tr_memberships, te_memberships, cluster_dict)
    print(f'input pipeline for {config.dataset} dataset is successfully built!')

    # --- make network model
    model = model_comparator.vgg_mdba(config)
    model.summary()
    write_log(log_file, f'Model name: {model.name}')
    try:
        model.load_weights(ckpt_file)
        write_log(log_file, f'Parameters are loaded from {ckpt_file}')
    except:
        write_log(log_file, f'Network are initialized with IMAGENET feature')

    # --- Optimizer, metrics, ckpt setting
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.base_lr,
                                                              config.lr_decay_steps,
                                                              config.lr_decay_rate,
                                                              staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss_metric = keras.metrics.Mean()
    test_loss_metric = keras.metrics.Mean()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, ckpt_path, max_to_keep=3)
    summary_writer = tf.summary.create_file_writer(summary_path)

    # --- Train / test loop
    best = 0.65
    for epoch in range(config.num_epoch):
        write_log(log_file, f'\n *** TRAIN - Epoch {epoch} *** ')
        # train one epoch
        for step, (base_img, ref_img, order_labels, _) in enumerate(train_data):
            preds, loss = train_one_step(model, optimizer, [base_img, ref_img], order_labels)

            train_loss_metric(loss)
            train_acc_metric(order_labels, preds)
            if step%200 == 0:
                write_log(log_file, f'{datetime.now().strftime("%Y%m%d-%H%M%S")} - EPOCH: {epoch},  GLOBAL_STEP: {int(checkpoint.step)},  STEP: {step},  '
                          f'ACC: {train_acc_metric.result()},  LOSS: {train_loss_metric.result()}')
            with summary_writer.as_default():
                tf.summary.scalar('train/loss', train_loss_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('train/accuracy', train_acc_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('learning_rate', optimizer.lr(int(checkpoint.step)).numpy(),
                                  step=int(checkpoint.step))

            checkpoint.step.assign_add(1)
            train_acc_metric.reset_states()
            train_loss_metric.reset_states()

        # test
        if epoch % config.test_freq == 0:
            total_preds = []
            total_gt = []
            for base_img, ref_img, order_labels, _ in test_data:
                preds, loss = test_one_step(model, [base_img, ref_img], order_labels)
                total_preds.append(preds)
                total_gt.append(order_labels)

                test_loss_metric(loss)
                test_acc_metric(order_labels, preds)

            write_log(log_file, f'\n *** TEST - Epoch {epoch} *** ')
            write_log(log_file, f'TEST-ACC: {test_acc_metric.result()},  TEST-LOSS: {test_loss_metric.result()}')

            with summary_writer.as_default():
                tf.summary.scalar('test/loss', test_loss_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('test/accuracy', test_acc_metric.result(), step=int(checkpoint.step))

            # --- Compute per class accuracy
            total_preds = np.concatenate(total_preds, axis=0)
            total_preds = np.argmax(total_preds, axis=-1)
            total_gt = np.concatenate(total_gt, axis=0)
            classes = np.unique(total_gt)

            for cls in classes:
                idx = np.argwhere(total_gt == cls)
                per_cls_acc = np.sum(total_preds[idx] == total_gt[idx]) / float(len(idx))
                write_log(log_file, f'ACC of cls {cls}: {per_cls_acc}   NUM_cases: {len(idx)}')

            # --- Save checkpoint
            if best < test_acc_metric.result():
                best = test_acc_metric.result()
                ckpt_name = os.path.join(ckpt_path, f'ckpt_{best}')
                model.save_weights(ckpt_name)
                print(f'Checkpoint saved: {ckpt_name}.')

                # AGE ESTIMATION

            test_acc_metric.reset_states()
            test_loss_metric.reset_states()







if __name__ == "__main__":
    main()
