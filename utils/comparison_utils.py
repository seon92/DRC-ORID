import numpy as np
import tensorflow as tf


# ==================================================================================================================== #
#                                                  compute order                                                       #
# ==================================================================================================================== #
#   --- functions for ternary order
def compute_ternary_order_tau(base_score, ref_score, tau):
    if base_score - ref_score > tau:
        order = 0
    elif abs(base_score - ref_score) <= tau:
        order = 1
    elif base_score - ref_score < -tau:
        order = 2
    else:
        raise ValueError(f'order relation is wrong. (base,ref,tau): {base_score, ref_score, tau}')
    return order


def compute_ternary_order_fixed_base(base_score, ref_scores, tau):
    # base score is fixed.
    num_scores = len(ref_scores)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_ternary_order_tau(base_score, ref_scores[idx], tau)
    return orders


def compute_ternary_order_fixed_ref(ref_score, base_scores, tau):
    # ref score is fixed.
    num_scores = len(base_scores)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_ternary_order_tau(base_scores[idx], ref_score, tau)
    return orders


#   --- functions for binary order
def compute_binary_order(base_score, ref_score):
    if base_score >= ref_score:
        order = 0
    elif base_score < ref_score:
        order = 1
    else:
        raise ValueError(f'order relation is wrong. (base,ref,tau): {base_score, ref_score}')
    return order


def compute_binary_order_fixed_base(base_score, ref_scores):
    # base score is fixed.
    num_scores = len(ref_scores)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_binary_order(base_score, ref_scores[idx])
    return orders


def compute_binary_order_fixed_ref(ref_score, base_scores):
    # ref score is fixed.
    num_scores = len(base_scores)
    orders = np.zeros((num_scores,), dtype=np.int32)
    for idx in range(num_scores):
        orders[idx] = compute_binary_order(base_scores[idx], ref_score)
    return orders


def find_reference_batch_in_log(base_scores, memberships, cluster_dict, labels, tau):
    """select ref and compute order label in log scale
    """
    ref_id_batch = np.zeros([len(base_scores),])
    order_batch = np.zeros([len(base_scores),])
    for i_in_batch, membership_id in enumerate(memberships):
        cands = cluster_dict[membership_id]
        ref_id = np.random.choice(cands, 1)[0]
        ref_id_batch[i_in_batch] = ref_id
        order_batch[i_in_batch] = compute_ternary_order_tau(np.log(base_scores[i_in_batch]), np.log(labels[ref_id]), tau)

    return ref_id_batch.astype(np.int32), order_batch


def find_reference_batch(base_scores, memberships, cluster_dict, labels, tau):
    """select ref and compute order label in log scale
    """
    ref_id_batch = np.zeros([len(base_scores),])
    order_batch = np.zeros([len(base_scores),])
    for i_in_batch, membership_id in enumerate(memberships):
        cands = cluster_dict[membership_id]
        ref_id = np.random.choice(cands, 1)[0]
        ref_id_batch[i_in_batch] = ref_id
        order_batch[i_in_batch] = compute_ternary_order_tau(base_scores[i_in_batch], labels[ref_id], tau)

    return ref_id_batch.astype(np.int32), order_batch


# ==================================================================================================================== #
#                                             estimation method (saaty, voting)                                        #
# ==================================================================================================================== #
def saaty_scaling_method(ref_scores, ratio_preds):
    # TL. Saaty, rf) https://www.sciencedirect.com/science/article/pii/0022249677900335
    num_ref = len(ref_scores)
    ref_scores = np.reshape(ref_scores, (-1, 1))
    zero_idx = np.where(ref_scores < 0.0001)
    ref_scores[zero_idx] = 0.004

    # make comparison matrix
    A = np.ones((num_ref+1, num_ref+1), dtype=np.float32)
    A[:num_ref, :num_ref] = ref_scores * (1/ref_scores.T)
    A[-1, :num_ref] = ratio_preds
    A[:num_ref, -1] = 1/ratio_preds
    col_sum_A = np.sum(A, axis=0)
    A = A / col_sum_A

    # Eigen value decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(A)

    # find maximum eigen value
    modulus = np.absolute(eigen_vals)
    max_idx = np.argmax(modulus)

    # compute kappa
    eigen_vector = eigen_vecs[:, max_idx]
    # eigen_vector = np.absolute(eigen_vecs[:, max_idx])
    u_ref, u = np.split(eigen_vector, [num_ref])
    kappa = np.sum(u_ref * np.squeeze(ref_scores)) / np.sum(u_ref * u_ref)

    # prediction by scaling
    error = np.mean(np.squeeze(ref_scores) - kappa * u_ref)
    pred_score = kappa * u

    return pred_score, error


def one_step_voting_ternary(orders, scores, tau, score_levels):
    num_refs = len(orders)
    votes = np.zeros_like(score_levels, dtype=np.int32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(score_levels > (scores[idx]+tau))[0, 0]
            except:
                min_idx = -1
            max_idx = len(score_levels)
        elif order == 1:
            try:
                min_idx = np.argwhere(score_levels >= (scores[idx]-tau))[0, 0]
            except:
                min_idx = -1

            max_idx = np.argwhere(score_levels <= (scores[idx]+tau))[-1, 0] + 1

        elif order == 2:
            min_idx = 0
            try:
                max_idx = np.argwhere(score_levels < (scores[idx]-tau))[-1, 0] + 1
            except:
                max_idx = 0
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        votes[min_idx:max_idx] += 1
    winners = np.argwhere(votes == np.amax(votes))
    # elected_index = winners[(len(winners)/2)][0]  # take the middle value when multiple winners exist.
    elected_index = winners[0][0]  # take the min value when multiple winners exist.
    return score_levels[elected_index], votes


def one_step_voting_ternary_log(orders, scores_in_log, tau, score_levels):
    num_refs = len(orders)
    votes = np.zeros_like(score_levels, dtype=np.int32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(score_levels > np.power(np.e, (scores_in_log[idx]+tau)))[0, 0]
            except:
                min_idx = -1
            max_idx = len(score_levels)
        elif order == 1:
            try:
                min_idx = np.argwhere(score_levels >= np.power(np.e, (scores_in_log[idx]-tau)))[0, 0]
            except:
                min_idx = -1
            max_idx = np.argwhere(score_levels <= np.power(np.e, (scores_in_log[idx]+tau)))[-1, 0] + 1

        elif order == 2:
            min_idx = 0
            try:
                max_idx = np.argwhere(score_levels < np.power(np.e, (scores_in_log[idx]-tau)))[-1, 0] + 1
            except:
                max_idx = 0
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        votes[min_idx:max_idx] += 1
    winners = np.argwhere(votes == np.amax(votes))
    # elected_index = winners[(len(winners)/2)][0]  # take the middle value when multiple winners exist.
    elected_index = winners[0][0]  # take the min value when multiple winners exist.
    return score_levels[elected_index], votes


def one_step_voting_binary(orders, scores, score_levels, tau=0.0):
    num_refs = len(orders)
    votes = np.zeros_like(score_levels, dtype=np.int32)
    votes_for_sum = np.zeros_like(score_levels, dtype=np.float32)

    for idx in range(num_refs):
        # compute where to vote
        order = orders[idx]
        if order == 0:
            try:
                min_idx = np.argwhere(score_levels >= (scores[idx]+tau))[0, 0]
            except:
                min_idx = -1
            max_idx = -1

        elif order == 1:
            min_idx = 0
            try:
                max_idx = np.argwhere(score_levels < (scores[idx]-tau))[-1, 0]
            except:
                max_idx = 0
        else:
            raise ValueError(f'order value is out of range: {order}')

        # voting
        votes[min_idx:max_idx] += 1
        votes_for_sum[min_idx:max_idx] += 1/(len(score_levels)-min_idx)
    winners = np.argwhere(votes == np.amax(votes))
    elected_index = winners[int((len(winners)/2))][0]  # take the middle value when multiple winners exist.
    # elected_index = winners[0][0]  # take the min value when multiple winners exist.
    return score_levels[elected_index], votes


def two_step_voting_ternary(orders, scores, tau):
    # rough estimation - first step
    rough_pred, _ = one_step_voting_ternary(orders, scores, tau, np.linspace(0, 1, 6))
    if rough_pred < tau:
        score_range = [0, 2*tau]
    elif rough_pred > 1-tau:
        score_range = [1-(2*tau), 1]
    else:
        score_range = [max(0, rough_pred-tau), min(rough_pred+tau, 1)]

    # select refs within score_range
    selected_idx = np.squeeze(np.argwhere(np.logical_and(scores >= score_range[0], scores <= score_range[1])))

    # find estimation - second step
    fine_pred, _ = one_step_voting_ternary(orders[selected_idx], scores[selected_idx], tau, np.linspace(score_range[0], score_range[1], 51))

    return fine_pred, rough_pred


def soft_voting_ternary(probs, scores, tau, score_levels):
    num_refs = len(probs)
    score_levels = score_levels.astype(np.float32)
    p_x = np.zeros_like(score_levels)

    for i_ref, ref_score in enumerate(scores):
        cond_p_per_levels = np.zeros((len(score_levels), 3))
        cond_probs = _conditional_probs_uniform_ternary(ref_score, tau, score_levels)
        order_per_levels = compute_ternary_order_fixed_ref(ref_score, score_levels, tau)
        for i_level, order in enumerate(order_per_levels):
            if order == 1:
                cond_p_per_levels[i_level, order] = cond_probs[order]
            else:
                cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += np.matmul(cond_p_per_levels, probs[i_ref])
    p_x = p_x / num_refs
    max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()

    # for binary classification : summation method
    low_scores = np.squeeze(np.argwhere(score_levels < 5.0))
    high_scores = np.squeeze(np.argwhere(score_levels >= 5.0))
    if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
        pred_by_sum = 0
    else:
        pred_by_sum = 1

    return score_levels[max_idx], np.sum(score_levels*p_x), pred_by_sum


def soft_voting_ternary_log(probs, scores, tau, score_levels):
    num_refs = len(probs)
    score_levels = score_levels.astype(np.float32)
    p_x = np.zeros_like(score_levels)

    for i_ref, ref_score in enumerate(scores):
        cond_p_per_levels = np.zeros((len(score_levels), 3))
        cond_probs = _conditional_probs_uniform_ternary_log(ref_score, tau, score_levels)
        order_per_levels = compute_ternary_order_fixed_ref(ref_score, np.log(score_levels), tau)
        for i_level, order in enumerate(order_per_levels):
            cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += np.matmul(cond_p_per_levels, probs[i_ref])
    p_x = p_x / num_refs
    max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()

    # for binary classification : summation method
    low_scores = np.squeeze(np.argwhere(score_levels < 5.0))
    high_scores = np.squeeze(np.argwhere(score_levels >= 5.0))
    if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
        pred_by_sum = 0
    else:
        pred_by_sum = 1

    return score_levels[max_idx], np.sum(score_levels*p_x), pred_by_sum


def tf_soft_voting_ternary(probs, scores, tau, score_levels):
    num_refs = probs.shape[0]
    score_levels = score_levels.astype(np.float32)
    p_x = tf.zeros_like(score_levels)

    for i_ref in range(num_refs):
        ref_score = scores[i_ref]
        cond_p_per_levels = np.zeros((len(score_levels), 3), dtype=np.float32)
        cond_probs = tf_conditional_probs_uniform_ternary(ref_score, tau, score_levels)
        order_per_levels = compute_ternary_order_fixed_ref(ref_score, score_levels, tau)
        for i_level, order in enumerate(order_per_levels):
            cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += tf.squeeze(tf.matmul(cond_p_per_levels, tf.expand_dims(probs[i_ref], axis=1)))
    p_x = p_x / num_refs

    return tf.math.reduce_sum(score_levels*p_x)


def tf_conditional_probs_uniform_ternary(ref_score, tau, score_levels):
    n_high = tf.where(score_levels > (ref_score+tau)).shape[0]
    n_low = tf.where(score_levels < (ref_score-tau)).shape[0]
    n_similar = tf.where(tf.math.logical_and(score_levels >= (ref_score-tau), score_levels <=(ref_score+tau))).shape[0]

    cond_probs = np.zeros((3,))
    for idx, n_levels in enumerate([n_high, n_similar, n_low]):
        if n_levels < 1:   # to prevent dividing by zero
            continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def soft_voting_binary(probs, scores, score_levels, tau=0.0):
    num_refs = len(probs)
    score_levels = score_levels.astype(np.float32)
    p_x = np.zeros_like(score_levels)

    for i_ref, ref_score in enumerate(scores):
        cond_p_per_levels = np.zeros((len(score_levels), 2))
        cond_probs = _conditional_probs_uniform_binary(ref_score, score_levels, tau=tau)
        order_per_levels = compute_binary_order_fixed_ref(ref_score, score_levels)
        for i_level, order in enumerate(order_per_levels):
            cond_p_per_levels[i_level, order] = cond_probs[order]
        p_x += np.matmul(cond_p_per_levels, probs[i_ref])

    # normalize the sum of probs to be 1.0
    p_x = p_x / num_refs
    max_idx = np.argmax(p_x)
    # plt.scatter(score_levels, p_x)
    # plt.xticks(score_levels)
    # plt.grid()

    # for binary classification : summation method
    low_scores = np.squeeze(np.argwhere(score_levels < 5.0))
    high_scores = np.squeeze(np.argwhere(score_levels >= 5.0))
    if np.sum(p_x[low_scores]) <= np.sum(p_x[high_scores]):
        pred_by_sum = 0
    else:
        pred_by_sum = 1

    return score_levels[max_idx], np.sum(score_levels*p_x), pred_by_sum


def _conditional_probs_uniform_ternary(ref_score, tau, score_levels, assertion=False):
    n_high = len(np.argwhere(score_levels > (ref_score+tau)))
    n_similar = len(np.argwhere(np.logical_and((ref_score-tau) <= score_levels, score_levels <= (ref_score+tau))))
    n_low = len(np.argwhere(score_levels < (ref_score-tau)))

    if assertion:
        assert((n_high + n_similar + n_low) == len(score_levels))

    cond_probs = np.zeros((3,))
    for idx, n_levels in enumerate([n_high, n_similar, n_low]):
        if n_levels < 1:   # to prevent dividing by zero
            continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def _conditional_probs_uniform_ternary_log(ref_score, tau, score_levels, assertion=False):

    n_high = len(np.argwhere(score_levels > np.power(np.e, (ref_score + tau))))
    n_similar = len(np.argwhere(np.logical_and(np.power(np.e, (ref_score - tau)) <= score_levels, score_levels <= np.power(np.e, (ref_score + tau)))))
    n_low = len(np.argwhere(score_levels < np.power(np.e, (ref_score - tau))))

    if assertion:
        assert((n_high + n_similar + n_low) == len(score_levels))

    cond_probs = np.zeros((3,))
    for idx, n_levels in enumerate([n_high, n_similar, n_low]):
        if n_levels < 1:   # to prevent dividing by zero
            continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def _conditional_probs_uniform_binary(ref_score, score_levels, tau=0.0):
    n_high = len(np.argwhere(score_levels >= (ref_score+tau)))
    n_low = len(np.argwhere(score_levels < (ref_score-tau)))

    cond_probs = np.zeros((2,))
    for idx, n_levels in enumerate([n_high, n_low]):
        if n_levels < 1:
            continue
        cond_probs[idx] = 1/n_levels
    return cond_probs


def order2ratio(preds, ratios):
    return ratios[preds]


# ==================================================================================================================== #
#                                                    quantization method                                               #
# ==================================================================================================================== #
def quantize_score_nearest(delta, x, score_range):
    num_step_func = int(np.floor(score_range / delta))
    quantized_score = np.zeros_like(x, dtype=np.float32)
    for i_step_func in range(num_step_func):
        shift = (i_step_func*delta) + (0.5*delta)
        quantized_score += (score_range/num_step_func) * _unit_step_function(x, shift)
    return quantized_score


def _unit_step_function(x, shift):
    return np.heaviside(x-shift, 1)


def quantize_scores_floor(delta, x, score_range):
    score_levels = int(score_range / delta) + 1
    quant_levels = np.linspace(0, score_range, score_levels)
    quantized_scores = np.zeros([len(x),])
    for i_x, score in enumerate(x):
        diff = quant_levels - score
        idx = np.argwhere(diff <= 0)
        idx = np.reshape(idx, [-1,])[-1]
        quantized_scores[i_x] = quant_levels[idx]
    return quantized_scores


def compute_cluster_index(score_levels, x, score_range):
    quant_levels = np.linspace(0, score_range, score_levels)
    diff = quant_levels - x
    idx = np.argwhere(diff <= 0)
    idx = np.reshape(idx, [-1, ])[-1]
    return idx


# ==================================================================================================================== #
#                                               reference search  method                                               #
# ==================================================================================================================== #
def select_reference_by_acc(scores, reliability, N):
    scores = np.squeeze(scores)
    classes, counts = np.unique(scores, return_counts=True)
    cls_idx = np.argwhere(counts >= N)
    classes = classes[np.squeeze(cls_idx)]
    ref_idx_list = np.zeros((len(classes), N), dtype=np.int32)

    for idx, score in enumerate(classes):
        ref_candidates = np.squeeze(np.argwhere(scores == score))
        acc_candidates = reliability[ref_candidates]
        top_n_idx = np.argsort(-acc_candidates)
        top_n_idx = top_n_idx[:N]
        ref_idx_list[idx, ...] = ref_candidates[top_n_idx]
    return ref_idx_list, classes


def knn_search(query_features, embedded_features, knn_k):
    # 1. compute distance
    distance_mat = _compute_distance_matrix(query_features, embedded_features)

    # 2. sort in ascending order
    sorting_idx = np.argsort(distance_mat, axis=-1)

    # 3. select nearest knn_k
    return sorting_idx[:, :knn_k]


def kfn_search(query_features, embedded_features, kfn_k):
    # 1. compute distance
    distance_mat = _compute_distance_matrix(query_features, embedded_features)

    # 2. sort in ascending order
    sorting_idx = np.argsort(distance_mat, axis=-1)

    # 3. select farthest k
    return sorting_idx[:, -kfn_k:]


def mix_search(query_features, embedded_features, k):
    # 1. compute distance
    distance_mat = _compute_distance_matrix(query_features, embedded_features)

    # 2. sort in ascending order
    sorting_idx = np.argsort(distance_mat, axis=-1)

    # 3. select closest and farthest samples
    return np.concatenate([sorting_idx[:, :int(k/2)], sorting_idx[:, -int(k/2):]], axis=-1)


def interval_search(query_features, embedded_features, k, interval=5):
    # 1. compute distance
    distance_mat = _compute_distance_matrix(query_features, embedded_features)

    # 2. sort in ascending order
    sorting_idx = np.argsort(distance_mat, axis=-1)

    # 3. select k samples with some interval
    return sorting_idx[:, np.arange(0, k*interval, step=interval)]


def _compute_distance_matrix(query_features, embedded_features):
    Q_square = np.sum(query_features * query_features, axis=-1, keepdims=True)
    E_square = np.sum(embedded_features * embedded_features, axis=-1, keepdims=True)
    QE = np.matmul(query_features, np.transpose(embedded_features))
    return Q_square - 2*QE + np.transpose(E_square)


if __name__ == "__main__":
    A = np.random.rand(100, 30)
    B = np.random.rand(50, 30)

    knn_idx = knn_search(A, B, 10)