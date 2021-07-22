import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# from utils.utils import load_images
from utils import load_images


def extract_normalized_attr_features(feature_extractor, filelist, config):
    if config.chain_feat_dim == -1:
        end_dim = config.z_dim
        feature_length = config.z_dim - config.age_feat_dim
    else:
        end_dim = config.age_feat_dim + config.chain_feat_dim
        feature_length = config.chain_feat_dim

    num_data = len(filelist)
    num_batch = np.ceil(num_data / config.batch_size).astype(np.int32)
    features = np.zeros([num_data, config.chain_feat_dim])
    for i_batch in range(num_batch):
        start_idx = i_batch * config.batch_size
        end_idx = min(start_idx + config.batch_size, num_data)
        if i_batch % 100 == 0:
            print(f'Extract features... {i_batch}/ {num_batch}')
        batch_image = load_images(filelist[start_idx:end_idx], config.width, config.height)
        feats = np.squeeze(feature_extractor(batch_image, training=False))
        feats = feats[:, config.age_feat_dim:end_dim]
        features[start_idx:end_idx] = normalize(np.reshape(feats, [-1, feature_length]))
    return features


def initial_kmeans_clustering(features, K):
    cluster_dict = dict()

    kmeans = KMeans(n_clusters=K).fit(features)
    centroids = kmeans.cluster_centers_
    centroids = normalize(centroids.astype(np.float32))
    memberships = kmeans.labels_

    for cluster_k in range(K):
        cluster_dict[cluster_k] = np.argwhere(memberships == cluster_k).flatten()

    return centroids, cluster_dict, memberships


def assign_memberships_by_nn_rule(features, centroids):
    similarities = compute_similarity(features, centroids)
    cluster_idx = np.argmax(similarities, axis=-1)
    return cluster_idx


def update_centroids_repulsive(K, features, memberships, alpha):
    centroids = np.zeros([K, features.shape[1]], dtype=np.float32)
    for k in range(K):
        idx = np.argwhere(memberships == k).flatten()
        idx_in_otr_clusters = np.argwhere(memberships != k).flatten()
        centroid = np.sum(features[idx], axis=0) - alpha*(1/(K-1))*np.sum(features[idx_in_otr_clusters], axis=0)
        centroids[k] = normalize(np.reshape(centroid, [1, -1])).flatten()
    return centroids


def track_membership(new_membership, old_membership, history, K):
    movement = np.zeros([K, K])
    num_data = new_membership.shape[0]
    for idx in range(num_data):
        movement[old_membership[idx], new_membership[idx]] += 1

    for cluster_k in range(K):
        history[cluster_k].append(movement[cluster_k, cluster_k] / np.sum(movement[:, cluster_k]))

    return movement, history


def measure_movement(K, new_memberships, old_memberships):
    movement = np.zeros([K, K], dtype=np.int32)
    num_data = new_memberships.shape[0]
    for idx in range(num_data):
        movement[old_memberships[idx], new_memberships[idx]] += 1

    change_ratio = 1 - (np.sum(np.diag(movement)) / np.sum(movement))
    return change_ratio, movement


def compute_similarity(features, centroids):
    """Compute cosine similarity between each feature and centroids
    Args:
        features: [N, C] l2 normalized data, cf) N: number of features, C: feature dimension
        centroids: [K, C] cluster centers

    Returns:
        similarites: [N, K]
    """
    return np.matmul(features, np.transpose(centroids, [1, 0]))


def regular_assign_repulsive(similarities, cluster_dict, memberships, regular_ratio=0.4, verbose=True):
    K = len(cluster_dict)
    num_data = similarities.shape[0]  # since bin label starts from 0
    stats_moved = 0

    distribution = np.zeros([K,])
    min_num_per_cluster = int(num_data * regular_ratio / K)

    for cluster_k in range(K):
        distribution[cluster_k] = len(cluster_dict[cluster_k])

    # determine to perform regular assign
    if not all(distribution > min_num_per_cluster):
        deficiencies = distribution - min_num_per_cluster
        for i_cluster in range(K):
            deficit = deficiencies[i_cluster]
            if deficit >= 0:
                continue
            to_add = []
            while deficit < 0:
                rich_cluster_idx = np.argwhere(deficiencies > 0).flatten()
                members_in_rich_clusters = np.concatenate([cluster_dict[cluster_idx] for cluster_idx in rich_cluster_idx])
                closest_idx = members_in_rich_clusters[np.argmax(similarities[members_in_rich_clusters, i_cluster])]   # global index
                og_membership = memberships[closest_idx]

                # delete
                cluster_dict[og_membership] = np.delete(cluster_dict[og_membership], np.argwhere(cluster_dict[og_membership] == closest_idx))
                deficiencies[og_membership] -= 1

                # add
                to_add.append(closest_idx)
                deficit += 1
                deficiencies[i_cluster] += 1
            cluster_dict[i_cluster] = np.concatenate([cluster_dict[i_cluster], np.array(to_add)])
            stats_moved += len(np.array(to_add))

    for cluster_k in range(K):
        member_idx = cluster_dict[cluster_k]
        memberships[member_idx] = cluster_k

    if verbose:
        print(f'NUM SAMPLES MOVED : {stats_moved}')

    return cluster_dict, memberships, stats_moved


def run_kmeans_repulsive(K, features, memberships, centroids, max_iter=70, stopping_threshold=0.01, min_cluster_ratio=0.5, alpha=0.5):
    """Run Kmeans
    Args:
        K (int): number of clusters
        features: [N, C] l2 normalized data, cf) N: number of features, C: feature dimension
        memberships (int): [N, ] current membership
        centroids: [K, C] cluster centers
        max_iter (int): maximum number of k-means iteration
        stopping_threshold: if membership change ratio is less than stopping threshold, stop running kmeans

    Returns:
        centroids: [K, C]
        memberships (int): [N, ]
    """
    N = features.shape[0]
    for i_iter in range(max_iter):
        old_memberships = memberships.copy()
        memberships = assign_memberships_by_nn_rule(features, centroids)
        centroids = update_centroids_repulsive(K, features, memberships, alpha=alpha)
        cluster_dict = dict()
        for cluster_k in range(K):
            cluster_dict[cluster_k] = np.argwhere(memberships==cluster_k).flatten()
        distances = compute_similarity(features, centroids)
        cluster_dict, memberships, stats_moved = regular_assign_repulsive(distances, cluster_dict, memberships, regular_ratio=min_cluster_ratio)
        if stats_moved > 0:
            centroids = update_centroids_repulsive(K, features, memberships, alpha=alpha)

        change_ratio, movement = measure_movement(K, memberships, old_memberships)
        # print(centroids)
        # print(movement)
        print(f'k-means iter{i_iter} : movement {change_ratio}')
        if change_ratio < stopping_threshold:
            print('early stopping')
            break

    return centroids, cluster_dict, memberships


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # toy example
    data = np.array([[0., -0.9],
                     [0., 0.2],
                     [0.3, 0.7],
                     [0.3, 0.1],
                     [1., -1.],
                     [-1.8, 1.1],
                     [1., 1.2],
                     [1.1, 1.]])
    data = normalize(data)
    memberships = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    centroids = normalize(np.array([[0.4, 0.6], [0.2, 0.2]]))
    # new_cents, cluster_dict, new_memberships = run_kmeans_repulsive(2, data, memberships, centroids)

    # K = 2
    # N = 8
    # color = ['blue', 'red', 'green']
    # for k in range(K):
    #     idx = np.argwhere(new_memberships==k).flatten()
    #     plt.scatter(data[idx,0], data[idx,1], c=color[k])
    #     plt.scatter(new_cents[k, 0], new_cents[k, 1], c=color[k], marker='*')

    # more complicate example
    N = 5000
    K = 6
    C = 2
    data = normalize(np.random.rand(N, C)-0.7)
    memberships = np.random.randint(0, K, N)
    centroids = normalize(np.random.rand(K, C))
    new_cents, cluster_dict, new_memberships = run_kmeans_repulsive(K, data, memberships, centroids, alpha=0.1)

    color = ['blue', 'red', 'green', 'purple', 'orange', 'pink']
    for k in range(K):
        idx = np.argwhere(new_memberships == k).flatten()
        plt.scatter(data[idx, 0], data[idx, 1], c=color[k])
        plt.scatter(new_cents[k, 0], new_cents[k, 1], c=color[k], marker='*', s=200)

    print('done')




