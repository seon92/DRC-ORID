import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d

# ==================================================================================================================== #
#                                            image visualization                                                       #
# ==================================================================================================================== #
def bgr2rgb(image):
    return image[..., ::-1]


def show_img(image, fig_size=(8, 8)):
    plt.figure(figsize=fig_size)
    plt.imshow(bgr2rgb(image))


# ==================================================================================================================== #
#                                            feature visualization                                                     #
# ==================================================================================================================== #
def label2color(label):
    # num color : 21
    colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'gold', 'tab:gray', 'tab:olive', 'tab:cyan',
              'navy', 'black', 'tab:pink', 'magenta', 'chocolate',
              'teal', 'slateblue', 'sandybrown', 'salmon', 'limegreen',
              'red', 'crimson', 'khaki', 'tomato', 'indigo',
              'forestgreen', 'darkviolet', 'orchid', 'saddlebrown', 'silver',
              'deeppink', 'darkslategrey', 'goldenrod', 'peru', 'seagreen',
              'aquamarine', 'wheat', 'gainsboro', 'rosybrown', 'palegreen', 'royalblue'
              ])
    return colors[label]


def chain_color(label):
    colors = np.array([[86/255, 168/255, 179/255], [255/255, 139/255, 139/255], [96/255, 96/255, 96/255], 'tab:olive',
                       'crimson', 'black', 'tab:purple'])
    return colors[label]


def tsne_visualize(features, labels, n_components=2, perplexity=40, n_iter=300, verbose=1):
    """
    :param features: input features
    :param labels: feature-wise label
    :param n_components: dimension of embedded space
    :return: TSNE visualization
    """
    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)

    classes = np.unique(labels)

    fig, ax = plt.subplots(figsize=(20, 20))
    for color_idx, cls in enumerate(classes):
        idx = np.argwhere(labels == cls)
        idx = np.squeeze(idx)
        cur_cls_feats = tsne_results[idx, ...]
        cur_cls_feats = np.reshape(cur_cls_feats, [-1, 2])
        ax.scatter(cur_cls_feats[:, 0], cur_cls_feats[:, 1], c=label2color(color_idx), label=cls, edgecolors='none')
    # ax.legend()
    # ax.grid(True)
    return fig, tsne_results


def tsne_visualize_two_labels(features, age_lb, membership_lb, n_components=2, perplexity=40, n_iter=300, verbose=1):
    """
    :param features: input features
    :param labels: feature-wise label
    :param n_components: dimension of embedded space
    :return: TSNE visualization
    """
    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)

    classes = np.unique(age_lb)

    fig, ax = plt.subplots(figsize=(20, 20))
    memberships = np.unique(membership_lb)
    for membership_id in memberships:
        cur_mem_idxs = np.argwhere(membership_id == membership_lb).flatten()
        cur_ages = age_lb[cur_mem_idxs]
        cur_feats = tsne_results[cur_mem_idxs]

        for color_idx, cls in enumerate(classes):
            idx = np.argwhere(cur_ages == cls)
            idx = np.squeeze(idx)
            cur_cls_feats = cur_feats[idx, ...]
            cur_cls_feats = np.reshape(cur_cls_feats, [-1, 2])
            ax.scatter(cur_cls_feats[:, 0], cur_cls_feats[:, 1], c=label2color(color_idx),
                       label=cls, marker=membership2marker(membership_id))
    # ax.legend()
    # ax.grid(True)
    return fig, tsne_results


def visualize_clusters(features, centroids, memberships, bin_labels, filename='dummy', n_components=2, perplexity=40, n_iter=300, verbose=1):
    # features = features.numpy()
    num_feat, f_dim = features.shape
    num_bin, K, _ = centroids.shape

    centroids_flat = np.reshape(centroids, [-1, f_dim])
    all_points = np.concatenate([features, centroids_flat])

    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(all_points)

    fig = plt.figure(figsize=(20, 20))

    bin_max = bin_labels.max()
    bin_min = bin_labels.min()

    age_color = (bin_labels - bin_min) / (bin_max - bin_min)

    for cluster_k in range(K):
        idx = np.argwhere(memberships == cluster_k).flatten()
        marker = membership2marker(cluster_k)
        colors = age_color[idx]
        points_xy = tsne_results[idx]
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, cmap=cm.rainbow, s=10, marker=marker)

    centroids_xy = tsne_results[num_feat:,:]
    centroids_xy = np.reshape(centroids_xy, [num_bin, K, 2])
    cent_color = np.arange(num_bin) / bin_max

    for cluster_k in range(K):
        points_xy = centroids_xy[:, cluster_k, :]
        marker = membership2marker(cluster_k)
        colors = cent_color
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, edgecolors='black', linewidths=3, cmap=cm.rainbow, s=250, marker=marker)

    h = [plt.plot([], [], color="gray", marker=membership2marker(i))[0] for i in range(K)]
    plt.legend(handles=h, labels=range(K), loc='lower right', title="Cluster")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
    return fig


def visualize_some_clusters(features, centroids, memberships, bin_labels, bin_id_to_visualize, filename='dummy', n_components=2, perplexity=40, n_iter=300, verbose=1):
    # features = features.numpy()
    num_feat, f_dim = features.shape
    num_bin, K, _ = centroids.shape

    centroids_flat = np.reshape(centroids, [-1, f_dim])
    all_points = np.concatenate([features, centroids_flat])

    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(all_points)

    fig = plt.figure(figsize=(20, 20))

    bin_max = bin_labels.max()
    bin_min = bin_labels.min()

    age_color = (bin_labels - bin_min) / (bin_max - bin_min)

    for cluster_k in range(K):
        idx = np.argwhere(memberships == cluster_k).flatten()
        marker = membership2marker(cluster_k)
        colors = age_color[idx]
        points_xy = tsne_results[idx]
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, cmap=cm.rainbow, s=10, marker=marker)

    centroids_xy = tsne_results[num_feat:,:]
    centroids_xy = np.reshape(centroids_xy, [num_bin, K, 2])
    cent_color = np.arange(num_bin) / bin_max

    for cluster_k in range(K):
        points_xy = centroids_xy[:, cluster_k, :]
        marker = membership2marker(cluster_k)
        colors = cent_color
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, edgecolors='black', linewidths=3, cmap=cm.rainbow, s=250, marker=marker)

    h = [plt.plot([], [], color="gray", marker=membership2marker(i))[0] for i in range(K)]
    plt.legend(handles=h, labels=range(K), loc='lower right', title="Cluster")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
    return fig


def visualize_clusters_attrs(features, centroids, memberships, bin_labels, genders, ages, filename='dummy', n_components=2, perplexity=40, n_iter=300, verbose=1):
    # features = features.numpy()
    num_feat, f_dim = features.shape
    num_bin, K, _ = centroids.shape

    centroids_flat = np.reshape(centroids, [-1, f_dim])
    all_points = np.concatenate([features, centroids_flat])

    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(all_points)

    fig = plt.figure(figsize=(20, 20))

    bin_max = bin_labels.max()
    bin_min = bin_labels.min()

    age_color = (bin_labels - bin_min) / (bin_max - bin_min)

    for cluster_k in range(K):
        idx = np.argwhere(memberships == cluster_k).flatten()
        marker = membership2marker(cluster_k)
        colors = age_color[idx]
        points_xy = tsne_results[idx]
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, cmap=cm.rainbow, s=10, marker=marker)

    centroids_xy = tsne_results[num_feat:,:]
    centroids_xy = np.reshape(centroids_xy, [num_bin, K, 2])
    cent_color = np.arange(num_bin) / bin_max

    for cluster_k in range(K):
        points_xy = centroids_xy[:, cluster_k, :]
        marker = membership2marker(cluster_k)
        colors = cent_color
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, edgecolors='black', linewidths=3, cmap=cm.rainbow, s=250, marker=marker)

    h = [plt.plot([], [], color="gray", marker=membership2marker(i))[0] for i in range(K)]
    plt.legend(handles=h, labels=range(K), loc='lower right', title="Cluster")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()
    return fig


def visualize_clusters_classification(features, centroids, labels, filename='dummy', n_components=2, perplexity=40, n_iter=300, verbose=1):
    # features = features.numpy()
    num_feat, f_dim = features.shape
    K, _ = centroids.shape

    all_points = np.concatenate([features, centroids])

    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(all_points)

    fig = plt.figure(figsize=(20, 20))

    for cluster_k in range(K):
        idx = np.argwhere(labels == cluster_k).flatten()
        colors = label2color(cluster_k)
        points_xy = tsne_results[idx]
        plt.scatter(points_xy[:, 0], points_xy[:, 1], c=colors, cmap=cm.rainbow, label=cluster_k, s=10)

    centroids_xy = tsne_results[num_feat:,:]

    for cluster_k in range(K):
        points_xy = centroids_xy[cluster_k]
        colors = label2color(cluster_k)
        plt.scatter(points_xy[0], points_xy[1], c=colors, edgecolors='black', linewidths=1, cmap=cm.rainbow, s=250, marker='*')

    plt.legend()
    plt.savefig(filename)
    plt.close()
    return fig


def age2color(age):
    max_age = age.max()
    min_age = age.min()
    normalized_age = (age - min_age) / (max_age - min_age)   # convert to [0, 1]
    uniq_age = np.unique(age)
    colors_for_bin = cm.rainbow((uniq_age - min_age) / (max_age - min_age))
    return cm.rainbow(normalized_age), colors_for_bin


def membership2marker(memberships):
    markers = np.array(['o', '^', 'v', 's', 'D', '>', 'p', '<'])
    return markers[memberships]


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


# 3D visualizer
def draw_pts(pts, title=None, clr=None, cmap=None, ax=None,sz=20, view_point=None):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        if view_point is not None:
            ax.view_init(view_point[0], view_point[1])
        else:
            ax.view_init(30,0)
    else:
        ax.cla()
    pts -= np.mean(pts,axis=0) #demean

    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if cmap is None and clr is not None:
        # assert(np.all(clr.shape==pts.shape))
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            # edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _,clr,_ = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            depthshade=True,
            norm=colors.Normalize(vmin=0, vmax=1)
            # edgecolors=(0.5, 0.5, 0.5)
        )

    ax.set_axis_off()
    ax.set_facecolor("white")
    if title is not None: ax.set_title(title)
    return fig
#


if __name__ == "__main__":
    ages = np.random.randint(3, 50, 20)
    c = age2color(ages)

    feats = np.random.rand(50, 100)
    centroids = np.random.rand(4, 2, 100)

    memberships = np.random.randint(0,2, 50)
    bin_labels =  np.random.randint(0,5, 50)
    bin_range = np.array([[10, 10], [11, 12], [13, 13], [14, 15], [16, 16]])
    f = visualize_clusters(feats, centroids, memberships, bin_labels)
    f.show()
    print('done')