import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from utils.finch import FINCH

def cluster_features(features, num_clusters, cluster_type='kmeans', affinity=None, smooth_kernel=None):

    if cluster_type == 'kmeans':
        cluster = KMeans(n_clusters=num_clusters, random_state=0, max_iter=10000).fit(features)
        cluster_labels = cluster.labels_
    elif cluster_type == 'spectral':
        if affinity is None:
            cluster = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0).fit(features)
        else:
            if smooth_kernel is not None:
                Kp = np.ones((smooth_kernel, smooth_kernel))/(smooth_kernel*smooth_kernel)
                affinity = fftconvolve(affinity, Kp, mode='same')
            cluster = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0, affinity='precomputed').fit(affinity)
        cluster_labels = cluster.labels_

    elif cluster_type == 'finch':
        c, num_clust, cluster_labels = FINCH(features, initial_rank=None, req_clust=num_clusters, tw_finch=False, distance='cosine', ensure_early_exit=True, verbose=False)
    elif cluster_type == 'twfinch':
        c, num_clust, cluster_labels = FINCH(features, initial_rank=None, req_clust=num_clusters, tw_finch=True, distance='cosine', ensure_early_exit=True, verbose=False)


    return cluster_labels


def get_hungarian_matching(cluster_labels, labels_gt, num_classes):
    
    assert(len(labels_gt) == len(cluster_labels))
    # Hungarian matching: swap the indices of clusters to maximize agreement with categorical
    num_elems = len(cluster_labels)
    match = _hungarian_match(cluster_labels, labels_gt, num_classes)

    reordered_preds = np.zeros(num_elems, dtype=cluster_labels.dtype)
    for pred_i, target_i in match:
        reordered_preds[cluster_labels == int(pred_i)] = int(target_i)

    return reordered_preds, match


def hungarian_evaluate(output, target, match):
    # Gather performance metrics
    assert(len(output) == len(target))
    mof = int((output == target).sum()) / float(len(target))
    iou = metrics.jaccard_score(target, output, average='weighted')
    nmi = metrics.normalized_mutual_info_score(target, output)
    ari = metrics.adjusted_rand_score(target, output)
    
    return mof, iou, nmi, ari

def compute_f1_score(output, target):
    f1_score = metrics.f1_score(target, output, average='macro')
    return f1_score

def compute_selfsimilarity(features):
    similarity_matrix = metrics.pairwise.cosine_similarity(features)
    return similarity_matrix


def _hungarian_match(flat_preds, flat_targets, num_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum()) 
            num_correct[c1, c2] = votes 

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    
    match = np.array(list(zip(*match)))
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def levenstein_distance(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col))
    else:
        score = D[-1, -1]

    return score

def estimate_cost_matrix(gt_labels, cluster_labels):
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_labels)
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, dim_1))
    for i in L_pred:
        idx = np.where(cluster_labels == i)
        gt_selected = gt_labels[idx]
        for j in L_gt:
            profit_mat[i][j] = np.count_nonzero(gt_selected == j)
    return -profit_mat

