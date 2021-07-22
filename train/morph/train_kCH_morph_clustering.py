import os
import pickle as pkl
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score as compute_nmi

sys.path.append('..')
from configs.cfg_morph_kCH import ConfigMorphV0 as Config
from networks import model_ae
from networks import model_discriminator
from utils.loss_util import compute_scc_loss_from_softmax, compute_mae_loss, compute_discriminator_loss, \
    compute_generator_loss, compute_sphercial_clustering_loss
from input_pipeline.morph_kCH_log import input_pipeline

from utils.clustering_utils_v6_alpha_repulsive import extract_normalized_attr_features
from utils.clustering_utils_v6_alpha_repulsive import initial_kmeans_clustering
from utils.clustering_utils_v6_alpha_repulsive import run_kmeans_repulsive
from utils.clustering_utils_v6_alpha_repulsive import measure_movement
from utils.clustering_utils_v6_alpha_repulsive import assign_memberships_by_nn_rule

from utils.utils import write_log
from utils.visualize_util import bgr2rgb


@tf.function
def train_one_step(model, discriminator, model_optimizer, disc_optimizer, data, order_labels, cluster_labels, centroids):
    base_img, ref_img = data
    losses = dict()

    with tf.GradientTape() as model_tape, tf.GradientTape() as disc_tape:
        gen_base, gen_ref, attr_base, attr_ref, order_forward, order_reverse = model(data)
        fake_base = discriminator(gen_base)
        fake_ref = discriminator(gen_ref)
        real_base = discriminator(base_img)
        real_ref = discriminator(ref_img)

        order_forward_loss = compute_scc_loss_from_softmax(order_labels, order_forward)
        order_reverse_loss = compute_scc_loss_from_softmax(order_forward.shape[-1] - 1 - order_labels, order_reverse)
        base_mae_loss = compute_mae_loss(base_img, gen_base)
        ref_mae_loss = compute_mae_loss(ref_img, gen_ref)
        gen_loss = compute_generator_loss(fake_base, fake_ref)
        if centroids is not None:
            base_cluster_loss = compute_sphercial_clustering_loss(centroids, attr_base, cluster_labels)
            ref_cluster_loss = compute_sphercial_clustering_loss(centroids, attr_ref, cluster_labels)
            loss = (order_forward_loss + order_reverse_loss) + 5 * (base_mae_loss + ref_mae_loss) + (gen_loss) \
                   + 0.1 * (base_cluster_loss + ref_cluster_loss)
        else:
            base_cluster_loss = base_mae_loss
            loss = (order_forward_loss + order_reverse_loss) + 5 * (base_mae_loss + ref_mae_loss) + (gen_loss)

        disc_loss = compute_discriminator_loss(real_base, real_ref, fake_base, fake_ref)

    model_gradients = model_tape.gradient(loss, model.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    model_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    losses['order_forward_loss'] = order_forward_loss
    losses['order_reverse_loss'] = order_reverse_loss
    losses['base_mae_loss'] = base_mae_loss
    losses['ref_mae_loss'] = ref_mae_loss
    losses['gen_loss'] = gen_loss
    losses['disc_loss'] = disc_loss
    losses['base_cluster_loss'] = base_cluster_loss
    losses['loss'] = loss

    return losses, gen_base, order_forward


@tf.function
def test_one_step(model, data, order_labels):
    base_img, ref_img = data
    gen_base, gen_ref, attr_base, attr_ref, order_forward, _ = model(data, training=False)
    order_loss = compute_scc_loss_from_softmax(order_labels, order_forward)
    base_mae_loss = compute_mae_loss(base_img, gen_base)
    ref_mae_loss = compute_mae_loss(ref_img, gen_ref)
    loss = order_loss + 0.5 * (base_mae_loss + ref_mae_loss)
    return loss, base_mae_loss, ref_mae_loss, order_loss, gen_base, order_forward


def main():
    def green(x):
        return '\033[92m' + x + '\033[0m'

    def blue(x):
        return '\033[94m' + x + '\033[0m'

    # --- select GPU to use
    GPU = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU  # specify which GPU(s) to be used
    print(f'USE GPU {GPU}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- load configs
    config = Config()
    experiment_name = f'Morph_drc_{config.dataset}_K_{config.K}_alpha_{config.alpha}_' + datetime.now().strftime(
        "%Y%m%d-%H%M%S")

    # --- generate log files
    log_path = f'results/results_morph/{experiment_name}'
    ckpt_path = f'{log_path}/checkpoints'
    ckpt_to_load = ''
    centroid_path = f'{log_path}/centroids'
    summary_path = f'{log_path}/summary'
    analysis_path = f'{log_path}/analysis'
    if not os.path.exists(log_path): os.mkdir(log_path)
    if not os.path.exists(ckpt_path): os.mkdir(ckpt_path)
    if not os.path.exists(summary_path): os.mkdir(summary_path)
    if not os.path.exists(analysis_path): os.mkdir(analysis_path)
    if not os.path.exists(centroid_path): os.mkdir(centroid_path)

    log_file = open(f'{log_path}/log.txt', 'w')
    nmi_file = open(f'{log_path}/nmi_log.txt', 'w')
    ca_file = open(f'{log_path}/ca_log.txt', 'w')

    # --- record the config of experiment
    config_dict = vars(Config)
    write_log(log_file, '*' * 100)
    for key in config_dict.keys():
        if not key.startswith('_'):
            write_log(log_file, f'{key} : {config_dict[key]}')
    write_log(log_file, '*' * 100 + '\n')

    # --- load data
    train_memberships = np.zeros([config.num_train, ], dtype=np.int32)
    cluster_dict = dict()
    cluster_dict[0] = np.arange(config.num_train)
    test_memberships = np.zeros([config.num_test, ], dtype=np.int32)
    centroids = None

    train_data = pd.read_csv(config.train_list, sep=" ")
    test_data = pd.read_csv(config.test_list, sep=" ")
    train_race = np.array(train_data['race'])
    test_race = np.array(test_data['race'])
    train_gender = np.array(train_data['gender'])
    test_gender = np.array(test_data['gender'])

    train_filelist = np.array(
        [f'{config.img_folder}/{train_data["database"][idx]}/{train_data["filename"][idx]}' for idx in
         range(len(train_data))])
    test_filelist = np.array(
        [f'{config.img_folder}/{test_data["database"][idx]}/{test_data["filename"][idx]}' for idx in
         range(len(test_data))])

    train_data, test_data = input_pipeline(config, train_memberships, test_memberships, cluster_dict)

    print(f'{config.dataset} dataset is successfully loaded!')

    # --- make network model
    model = model_ae.spherical_v0(config)
    model.summary()

    encoder = model.get_layer('encoder')
    encoder = keras.Model(inputs=encoder.input, outputs=encoder.output)

    discriminator = model_discriminator.discriminator_v3()

    write_log(log_file, f'Model name: {model.name}')
    try:
        model.load_weights(ckpt_to_load)
        write_log(log_file, f'Parameters are loaded from {ckpt_to_load}')
    except:
        write_log(log_file, f'Network are initialized with Random variables')

    # --- Optimizer, metrics, ckpt setting
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.base_lr,
                                                              config.lr_decay_steps,
                                                              config.lr_decay_rate,
                                                              staircase=True)
    model_optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    disc_optimizer = keras.optimizers.Adam(learning_rate=config.base_lr)
    train_loss_metric = keras.metrics.Mean()
    test_order_loss_metric = keras.metrics.Mean()
    test_mae_loss_metric = keras.metrics.Mean()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    summary_writer = tf.summary.create_file_writer(summary_path)

    # --- Train / test loop
    best = 0.70

    for epoch in range(config.num_epoch):
        print(f'\n *** {blue("TRAIN")} - Epoch {epoch} *** ')
        # train one epoch
        for step, (base_img, ref_img, order_labels, cluster_labels) in enumerate(train_data):
            losses, gen_base, pred_forward = \
                train_one_step(
                    model, discriminator,
                    model_optimizer,
                    disc_optimizer,
                    [base_img, ref_img],
                    order_labels, cluster_labels, centroids
                )
            train_loss_metric(losses['loss'])
            train_acc_metric(order_labels, pred_forward)
            if step % 50 == 0:
                print(
                    f'{blue("Train")} {datetime.now().strftime("%Y%m%d-%H%M%S")} EPOCH: {epoch},  GLOBAL_STEP: {int(checkpoint.step)},  STEP: {step},  '
                    f'ACC: {train_acc_metric.result():.4f},  '
                    f'Base-MAE: {losses["base_mae_loss"].numpy():.4f},  Ref-MAE: {losses["ref_mae_loss"].numpy():.4f}  '
                    f'Order f-loss : {losses["order_forward_loss"].numpy():.2f},  Order r-loss : {losses["order_reverse_loss"].numpy():.4f}  '
                    f'Gen-loss : {losses["gen_loss"].numpy():.4f},  Disc-loss : {losses["disc_loss"].numpy():.4f}  base_cluster_loss: {losses["base_cluster_loss"].numpy():.4f}  '
                    f'LOSS: {train_loss_metric.result():.4f}')

            if step % 2000 == 0:
                plt.imsave(f'{analysis_path}/ep{epoch}_step{step}_gen.png', np.clip(bgr2rgb(gen_base[0].numpy()), 0, 1))
                plt.imsave(f'{analysis_path}/ep{epoch}_step{step}_gt.png', np.clip(bgr2rgb(base_img[0].numpy()), 0, 1))

            with summary_writer.as_default():
                tf.summary.scalar('train/loss', train_loss_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('train/base_mae_loss', losses["base_mae_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/ref_mae_loss', losses["ref_mae_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/order_f-loss', losses["order_forward_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/order_r-loss', losses["order_reverse_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/cluster-loss', losses["base_cluster_loss"], step=int(checkpoint.step))
                tf.summary.scalar('train/gen-loss', losses["gen_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/disc-loss', losses["disc_loss"].numpy(), step=int(checkpoint.step))
                tf.summary.scalar('train/order_accuracy', train_acc_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('learning_rate', model_optimizer.lr(int(checkpoint.step)).numpy(),
                                  step=int(checkpoint.step))

            checkpoint.step.assign_add(1)
            train_acc_metric.reset_states()
            train_loss_metric.reset_states()

        train_nmi = 0
        # --- rebuild input pipeline
        if config.K > 1:
            if epoch == 2:
                train_attr_features = extract_normalized_attr_features(encoder, train_filelist, config)
                test_attr_features = extract_normalized_attr_features(encoder, test_filelist, config)

                old_memberships = train_memberships.copy()
                centroids, cluster_dict, train_memberships = initial_kmeans_clustering(train_attr_features, config.K)
                change_ratio, movement = measure_movement(config.K, train_memberships, old_memberships)
                print(f'membership change ratio: {change_ratio}')
                print(movement)
                test_memberships = assign_memberships_by_nn_rule(test_attr_features, centroids)
                print('\n')
                write_log(log_file, blue('TRAIN'))
                for k in range(config.K):
                    cur_cluster_idx = np.argwhere(train_memberships == k).flatten()
                    cur_race = train_race[cur_cluster_idx]
                    cur_cluster_size = len(cur_cluster_idx)
                    write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                    write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                    cur_gender = train_gender[cur_cluster_idx]
                    write_log(log_file,
                              f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                    print('\n')

                write_log(log_file, green('TEST'))
                for k in range(config.K):
                    cur_cluster_idx = np.argwhere(test_memberships == k).flatten()
                    cur_race = test_race[cur_cluster_idx]
                    cur_cluster_size = len(cur_cluster_idx)
                    write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                    write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                    cur_gender = test_gender[cur_cluster_idx]
                    write_log(log_file,
                              f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                    print('\n')

                train_data, test_data = input_pipeline(config, train_memberships, test_memberships,
                                                       cluster_dict)
                print('input pipeline is rebuilt with updated clustering information!')
                train_nmi = compute_nmi(train_memberships, train_race)
                test_nmi = compute_nmi(test_memberships, test_race)
                print('** centroid affinity')
                write_log(ca_file, f'{np.dot(centroids[0], centroids[1])}')
                print('** NMI')
                write_log(nmi_file, f'{train_nmi} {test_nmi}')

            elif (epoch > 2) and (epoch < 200):
                if epoch % 3 == 0:
                    train_attr_features = extract_normalized_attr_features(encoder, train_filelist, config)
                    test_attr_features = extract_normalized_attr_features(encoder, test_filelist, config)

                    old_memberships = train_memberships.copy()
                    centroids, cluster_dict, train_memberships = run_kmeans_repulsive(config.K,
                                                                                      train_attr_features,
                                                                                      train_memberships,
                                                                                      centroids,
                                                                                      alpha=config.alpha)

                    change_ratio, movement = measure_movement(config.K, train_memberships, old_memberships)
                    print(f'membership change ratio: {change_ratio}')
                    print(movement)
                    test_memberships = assign_memberships_by_nn_rule(test_attr_features, centroids)
                    print('\n')
                    write_log(log_file, blue('TRAIN'))
                    for k in range(config.K):
                        cur_cluster_idx = np.argwhere(train_memberships == k).flatten()
                        cur_race = train_race[cur_cluster_idx]
                        cur_cluster_size = len(cur_cluster_idx)
                        write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                        write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                        cur_gender = train_gender[cur_cluster_idx]
                        write_log(log_file,
                                  f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                        print('\n')

                    write_log(log_file, green('TEST'))
                    for k in range(config.K):
                        cur_cluster_idx = np.argwhere(test_memberships == k).flatten()
                        cur_race = test_race[cur_cluster_idx]
                        cur_cluster_size = len(cur_cluster_idx)
                        write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                        write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                        cur_gender = test_gender[cur_cluster_idx]
                        write_log(log_file,
                                  f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                        print('\n')

                    train_data, test_data = input_pipeline(config, train_memberships, test_memberships,
                                                           cluster_dict)

                    print('input pipeline is rebuilt with updated clustering information!')

                    train_nmi = compute_nmi(train_memberships, train_race)
                    test_nmi = compute_nmi(test_memberships, test_race)
                    print('** centroid affinity')
                    write_log(ca_file, f'{np.dot(centroids[0], centroids[1])}')
                    print('** NMI')
                    write_log(nmi_file, f'{train_nmi} {test_nmi}')

            elif epoch > 200:
                if epoch % 3 == 0:
                    train_attr_features = extract_normalized_attr_features(encoder, train_filelist, config)
                    test_attr_features = extract_normalized_attr_features(encoder, test_filelist, config)

                    old_memberships = train_memberships.copy()
                    centroids, cluster_dict, train_memberships = run_kmeans_repulsive(config.K,
                                                                                      train_attr_features,
                                                                                      train_memberships,
                                                                                      centroids,
                                                                                      alpha=0.5 * config.alpha)

                    change_ratio, movement = measure_movement(config.K, train_memberships, old_memberships)
                    print(f'membership change ratio: {change_ratio}')
                    print(movement)
                    test_memberships = assign_memberships_by_nn_rule(test_attr_features, centroids)
                    print('\n')
                    write_log(log_file, blue('TRAIN'))
                    for k in range(config.K):
                        cur_cluster_idx = np.argwhere(train_memberships == k).flatten()
                        cur_race = train_race[cur_cluster_idx]
                        cur_cluster_size = len(cur_cluster_idx)
                        write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                        write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                        cur_gender = train_gender[cur_cluster_idx]
                        write_log(log_file,
                                  f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                        print('\n')

                    write_log(log_file, green('TEST'))
                    for k in range(config.K):
                        cur_cluster_idx = np.argwhere(test_memberships == k).flatten()
                        cur_race = test_race[cur_cluster_idx]
                        cur_cluster_size = len(cur_cluster_idx)
                        write_log(log_file, f'cluster {k} :  {cur_cluster_size}')
                        write_log(log_file, f'race : 0: {cur_cluster_size - np.sum(cur_race)}   1: {np.sum(cur_race)}')
                        cur_gender = test_gender[cur_cluster_idx]
                        write_log(log_file,
                                  f'gender : 0: {cur_cluster_size - np.sum(cur_gender)}   1: {np.sum(cur_gender)}')
                        print('\n')

                    train_data, test_data = input_pipeline(config, train_memberships, test_memberships,
                                                           cluster_dict)
                    print('input pipeline is rebuilt with updated clustering information!')

                    train_nmi = compute_nmi(train_memberships, train_race)
                    test_nmi = compute_nmi(test_memberships, test_race)
                    print('** centroid affinity')
                    write_log(ca_file, f'{np.dot(centroids[0], centroids[1])}')
                    print('** NMI')
                    write_log(nmi_file, f'{train_nmi} {test_nmi}')

            if train_nmi > 0.9:
                ckpt_name = os.path.join(ckpt_path, f'ckpt_ep{epoch}')
                model.save_weights(ckpt_name)
                print(f'Checkpoint saved: {ckpt_name}.')
                centroid_name = f'{centroid_path}/centroids_ep{epoch}'
                pkl.dump(centroids, open(centroid_name, "wb"))
            # elif epoch == 2:
            #     ckpt_name = os.path.join(ckpt_path, f'ckpt_ep{epoch}')
            #     model.save_weights(ckpt_name)
            #     print(f'Checkpoint saved: {ckpt_name}.')
            #     centroid_name = f'{centroid_path}/centroids_ep{epoch}'
            #     pkl.dump(centroids, open(centroid_name, "wb"))
            # elif epoch % 3 == 0:
            #     ckpt_name = os.path.join(ckpt_path, f'ckpt_ep{epoch}')
            #     model.save_weights(ckpt_name)
            #     print(f'Checkpoint saved: {ckpt_name}.')
            #     centroid_name = f'{centroid_path}/centroids_ep{epoch}'
            #     pkl.dump(centroids, open(centroid_name, "wb"))

        # test
        if epoch % config.test_freq == 0:
            total_preds = []
            total_gt = []
            for step, (base_img, ref_img, order_labels, _) in enumerate(test_data):
                loss, base_mae_loss, ref_mae_loss, order_loss, gen_base, pred_forward = test_one_step(model, [base_img,
                                                                                                              ref_img],
                                                                                                      order_labels)
                total_preds.append(pred_forward)
                total_gt.append(order_labels)
                test_acc_metric(order_labels, pred_forward)
                test_order_loss_metric(order_loss)
                test_mae_loss_metric(base_mae_loss)

                if step == 0:
                    plt.imsave(f'{analysis_path}/test_ep{epoch}_gen.png', np.clip(bgr2rgb(gen_base[0].numpy()), 0, 1))
                    plt.imsave(f'{analysis_path}/test_ep{epoch}_gt.png', np.clip(bgr2rgb(base_img[0].numpy()), 0, 1))

            write_log(log_file, f'\n *** {green("TEST")} - Epoch {epoch} *** ')
            write_log(log_file,
                      f'TEST-ACC: {test_acc_metric.result():.4f},  TEST-ORDERLOSS: {test_order_loss_metric.result():.4f}'
                      f'   TEST-MAE: {test_mae_loss_metric.result():.4f}')

            with summary_writer.as_default():
                tf.summary.scalar('test/order_loss', test_order_loss_metric.result(), step=int(checkpoint.step))
                tf.summary.scalar('test/mae_loss', test_mae_loss_metric.result(), step=int(checkpoint.step))
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
                ckpt_name = os.path.join(ckpt_path, f'ckpt_ep{epoch}_{best}')
                model.save_weights(ckpt_name)
                print(f'Checkpoint saved: {ckpt_name}.')
                centroid_name = f'{centroid_path}/centroids_ep{epoch}'
                pkl.dump(centroids, open(centroid_name, "wb"))

            test_acc_metric.reset_states()
            test_order_loss_metric.reset_states()
            test_mae_loss_metric.reset_states()


if __name__ == "__main__":
    main()