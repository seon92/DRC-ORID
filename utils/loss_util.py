import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_addons as tfa
import numpy as np


def compute_scc_loss(labels, preds):
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, preds)


def compute_scc_loss_from_softmax(labels, preds, sample_weight=None):
    return keras.losses.SparseCategoricalCrossentropy()(labels, preds, sample_weight=sample_weight)


def compute_cc_loss_from_softmax(labels, preds, label_smoothing=0):
    return keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)(labels, preds)


def compute_cc_loss(labels, preds, label_smoothing=0):
    return keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing, from_logits=True)(labels, preds)


def compute_mae_loss(labels, preds):
    return keras.losses.MeanAbsoluteError()(labels, preds)


def compute_clustering_loss(centroids, features, batch_memberships, batch_bin_labels, alpha):
    num_bins, K, fdim = centroids.shape
    dummy = np.zeros([fdim,])
    pdist_list = []
    ndist_list = []
    for feature, bin_label, membership in zip(features, batch_bin_labels, batch_memberships):

        if bin_label == 0:
            cent_n = centroids[bin_label]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], dummy, cent_nxt[membership], [0, alpha])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], dummy, cent_nxt[neg_memberships], [0, alpha])

        elif bin_label == (num_bins-1):
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label - 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], cent_bf[membership], dummy, [alpha, 0])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], cent_bf[neg_memberships], dummy, [alpha, 0])

        else:
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label-1]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], cent_bf[membership], cent_nxt[membership], [alpha, alpha])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], cent_bf[neg_memberships], cent_nxt[neg_memberships], [alpha, alpha])

        pdist_list.append(pos_dist)
        ndist_list.append(neg_dist)

    loss = tf.math.reduce_sum(pdist_list) - tf.math.reduce_sum(ndist_list)
    return loss


def tf_compute_triplet_distance(feature, cent_n, cent_bf, cent_nxt, weight):
    return tf.math.reduce_sum(tf.norm(feature - cent_n, axis=-1) + weight[0]*tf.norm(feature - cent_bf, axis=-1) + weight[1]*tf.norm(feature - cent_nxt, axis=-1))


def compute_triplet_loss(centroids, features, batch_memberships, batch_bin_labels, alpha, rambda=1):
    num_bins, K, fdim = centroids.shape
    dummy = np.zeros([fdim,])
    batch_size = features.shape[0]
    features = tf.math.l2_normalize(features, axis=-1)
    loss = tf.zeros([1])
    for feature, bin_label, membership in zip(features, batch_bin_labels, batch_memberships):

        if bin_label == 0:
            cent_n = centroids[bin_label]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], dummy, cent_nxt[membership], [0, alpha])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], dummy, cent_nxt[neg_memberships], [0, alpha])

        elif bin_label == (num_bins-1):
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label - 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], cent_bf[membership], dummy, [alpha, 0])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], cent_bf[neg_memberships], dummy, [alpha, 0])

        else:
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label-1]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            pos_dist = tf_compute_triplet_distance(feature, cent_n[membership], cent_bf[membership], cent_nxt[membership], [alpha, alpha])
            neg_dist = tf_compute_triplet_distance(feature, cent_n[neg_memberships], cent_bf[neg_memberships], cent_nxt[neg_memberships], [alpha, alpha])

        margin = tf.math.maximum(pos_dist - neg_dist + rambda, 0)
        loss += margin
    loss = loss / batch_size
    return loss


def compute_clustering_loss_repulsive(centroids, features, batch_memberships, batch_bin_labels, alpha):
    num_bins, K, fdim = centroids.shape
    dummy = np.zeros([fdim,], dtype=np.float32)
    batch_size = features.shape[0]
    features = tf.math.l2_normalize(features, axis=-1)
    loss = tf.zeros([1])
    for feature, bin_label, membership in zip(features, batch_bin_labels, batch_memberships):

        if bin_label == 0:
            cent_n = centroids[bin_label]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf.math.subtract(1.5, tf_compute_cosine_similarity(feature, cent_n[membership], dummy, cent_nxt[membership], [0, alpha]))
            rep_loss = tf.math.add(1.5, tf_compute_cosine_similarity(feature, cent_n[neg_memberships], dummy, cent_nxt[neg_memberships], [0, alpha]))

        elif bin_label == (num_bins-1):
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label - 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf.math.subtract(1.5, tf_compute_cosine_similarity(feature, cent_n[membership], cent_bf[membership], dummy, [alpha, 0]))
            rep_loss = tf.math.add(1.5, tf_compute_cosine_similarity(feature, cent_n[neg_memberships], cent_bf[neg_memberships], dummy, [alpha, 0]))

        else:
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label-1]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf.math.subtract(2, tf_compute_cosine_similarity(feature, cent_n[membership], cent_bf[membership], cent_nxt[membership], [alpha, alpha]))
            rep_loss = tf.math.add(2, tf_compute_cosine_similarity(feature, cent_n[neg_memberships], cent_bf[neg_memberships], cent_nxt[neg_memberships], [alpha, alpha]))

        loss += (rep_loss + att_loss)  # minimize repulsive power and maximize attraction power
    loss = loss / batch_size
    return loss


def compute_clustering_loss_repulsive_v2(centroids, features, batch_memberships, batch_bin_labels, alpha):
    num_bins, K, fdim = centroids.shape
    dummy = np.zeros([fdim,], dtype=np.float32)
    batch_size = features.shape[0]
    loss = tf.zeros([1])
    for feature, bin_label, membership in zip(features, batch_bin_labels, batch_memberships):
        if bin_label == 0:
            cent_n = centroids[bin_label]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf_compute_cosine_similarity(feature, cent_n[membership], dummy, cent_nxt[membership], [0, alpha])
            rep_loss = tf_compute_cosine_similarity(feature, cent_n[neg_memberships], dummy, cent_nxt[neg_memberships], [0, alpha])

        elif bin_label == (num_bins-1):
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label - 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf_compute_cosine_similarity(feature, cent_n[membership], cent_bf[membership], dummy, [alpha, 0])
            rep_loss = tf_compute_cosine_similarity(feature, cent_n[neg_memberships], cent_bf[neg_memberships], dummy, [alpha, 0])

        else:
            cent_n = centroids[bin_label]
            cent_bf = centroids[bin_label-1]
            cent_nxt = centroids[bin_label + 1]
            neg_memberships = np.delete(np.arange(K), membership)
            att_loss = tf_compute_cosine_similarity(feature, cent_n[membership], cent_bf[membership], cent_nxt[membership], [alpha, alpha])
            rep_loss = tf_compute_cosine_similarity(feature, cent_n[neg_memberships], cent_bf[neg_memberships], cent_nxt[neg_memberships], [alpha, alpha])

        loss += (rep_loss - att_loss)  # minimize repulsive power and maximize attraction power
    loss = loss / batch_size
    return loss


def tf_compute_cosine_similarity(feature, cent_n, cent_bf, cent_nxt, weight):
    fdim = feature.shape[0]
    feature = tf.reshape(feature, [-1, 1])
    cent_n = tf.reshape(cent_n, [-1, fdim])
    cent_bf = tf.reshape(cent_bf, [-1, fdim])
    cent_nxt = tf.reshape(cent_nxt, [-1, fdim])
    return tf.math.reduce_mean(tf.matmul(cent_n, feature) + weight[0]*tf.matmul(cent_bf, feature) + weight[1]*tf.matmul(cent_nxt, feature))


def tf_compute_cosine_similarity_for_classification(feature, centroid):
    fdim = feature.shape[0]
    feature = tf.reshape(feature, [-1, 1])
    centroid = tf.reshape(centroid, [-1, fdim])
    return tf.math.reduce_mean(tf.matmul(centroid, feature))


def compute_clustering_loss_repulsive_classification(centroids, batch_features, batch_labels):
    K, fdim = centroids.shape
    batch_size = batch_features.shape[0]
    loss = tf.zeros([1])
    for feature, cls_label in zip(batch_features, batch_labels):
        neg_cls = np.delete(np.arange(K), cls_label)
        att_loss = tf_compute_cosine_similarity_for_classification(feature, centroids[cls_label])
        rep_loss = tf_compute_cosine_similarity_for_classification(feature, centroids[neg_cls])
        loss += (rep_loss - att_loss)  # minimize repulsive power and maximize attraction power
    loss = loss / batch_size
    return loss


def compute_margin_loss(base_feat, ref_feat, anchor, margin):
    return tf.reduce_mean(tf.maximum(tf.squeeze(tf.matmul(ref_feat, anchor) - tf.matmul(base_feat, anchor)) + margin, 0))


def compute_discriminator_loss(real_base, real_ref, fake_base, fake_ref, smoothing=0):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=smoothing)
    real_loss = cross_entropy(tf.ones_like(real_base), real_base) + cross_entropy(tf.ones_like(real_ref), real_ref)
    fake_loss = cross_entropy(tf.zeros_like(fake_base), fake_base) + cross_entropy(tf.zeros_like(fake_ref), fake_ref)
    disc_loss = real_loss + fake_loss
    return disc_loss


def compute_discriminator_loss_triple(real_base, real_ref, fake_base, fake_ref, fake_swap):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_base), real_base) + cross_entropy(tf.ones_like(real_ref), real_ref)
    fake_loss = cross_entropy(tf.zeros_like(fake_base), fake_base) + cross_entropy(tf.zeros_like(fake_ref), fake_ref) + cross_entropy(tf.zeros_like(fake_swap), fake_swap)
    disc_loss = real_loss + (fake_loss*2/3)
    return disc_loss


def compute_generator_loss(fake_base, fake_ref):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    fake_loss = cross_entropy(tf.ones_like(fake_base), fake_base) + cross_entropy(tf.ones_like(fake_ref), fake_ref)
    return fake_loss


def compute_generator_loss_triple(fake_base, fake_ref, fake_swap):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    fake_loss = cross_entropy(tf.ones_like(fake_base), fake_base) + cross_entropy(tf.ones_like(fake_ref), fake_ref) +  cross_entropy(tf.ones_like(fake_swap), fake_swap)
    return fake_loss


def compute_repulsive_chain_loss(centroids, features, batch_memberships, alpha=1):
    """Compute repulsive chain loss
        Args:
            features: [B, C] l2 normalized data, cf) B: batch size, C: feature dimension
            memberships (int): [B, ] current membership
            centroids: [K, C] cluster centers

        Returns:
            loss
        """
    batch_size = features.shape[0]
    K = centroids.shape[0]
    similarities = tf.matmul(features, tf.transpose(centroids, [1, 0]))   # [B, K]
    pos_idx = tf.concat([tf.expand_dims(tf.range(batch_size), axis=-1), tf.expand_dims(batch_memberships, axis=-1)], axis=-1)
    pos_similarities =  tf.gather_nd(similarities, pos_idx)  # [B, ]
    att_loss = tf.reduce_mean(pos_similarities)
    rep_loss = tf.reduce_mean(tf.reduce_sum(similarities, axis=-1) - pos_similarities)
    loss = alpha*(1/(K-1))*rep_loss - att_loss

    return loss


def compute_sphercial_clustering_loss(centroids, features, batch_memberships):
    """Compute repulsive chain loss
        Args:
            features: [B, C] l2 normalized data, cf) B: batch size, C: feature dimension
            memberships (int): [B, ] current membership
            centroids: [K, C] cluster centers

        Returns:
            loss
        """
    batch_size = features.shape[0]
    similarities = tf.matmul(features, tf.transpose(centroids, [1, 0]))   # [B, K]
    pos_idx = tf.concat([tf.expand_dims(tf.range(batch_size), axis=-1), tf.expand_dims(batch_memberships, axis=-1)], axis=-1)
    pos_similarities =  tf.gather_nd(similarities, pos_idx)  # [B, ]
    loss = -tf.reduce_mean(pos_similarities)

    return loss


def compute_repulsive_chain_loss_v2(centroids, features, batch_memberships):
    """Compute repulsive chain loss
        Args:
            features: [B, C] l2 normalized data, cf) B: batch size, C: feature dimension
            memberships (int): [B, ] current membership
            centroids: [K, C] cluster centers

        Returns:
            loss
        """
    batch_size = features.shape[0]
    K = centroids.shape[0]
    similarities = tf.matmul(features, tf.transpose(centroids, [1, 0]))   # [B, K]
    pos_idx = tf.concat([tf.expand_dims(tf.range(batch_size), axis=-1), tf.expand_dims(batch_memberships, axis=-1)], axis=-1)
    pos_similarities =  tf.gather_nd(similarities, pos_idx)  # [B, ]
    att_loss = tf.reduce_mean(pos_similarities)
    rep_loss = tf.reduce_mean(tf.reduce_sum(similarities, axis=-1) - pos_similarities)
    loss = (1/(K-1))*rep_loss - att_loss

    return loss


def compute_nt_xent_loss(base_features, ref_features, batch_memberships, temperature):
    batch_size = base_features.shape[0]
    similarities = tf.matmul(base_features, tf.transpose(ref_features, [1, 0]))  # [B, B]
    pos_sim = tf.linalg.diag_part(similarities) / temperature


    neg_idx = tf.concat([tf.expand_dims(tf.range(batch_size), axis=-1), tf.expand_dims(batch_memberships, axis=-1)],
                        axis=-1)
    pos_similarities = tf.gather_nd(similarities, pos_idx)  # [B, ]