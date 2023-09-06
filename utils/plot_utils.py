import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def get_topk_table(indicies, labels):
    topk_string = []
    for sample in indicies:
        list_topk = []
        for neighbor in sample:
            list_topk.append(labels[neighbor])
        topk_string.append(list_topk)

    return topk_string

def compute_tsne(features):
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2).fit_transform(features)
    return X_embedded


def save_tsne(tsne, labels, tsne_path):
    
    plt.figure()
    cmap = plt.cm.get_cmap('hsv')
    skip_sample = 2
    

    # Plot each data point
    for i, feature in enumerate(tsne[::skip_sample]):
        plt.scatter(feature[0], feature[1],
            s = 10,
            c = np.array([cmap(labels[i*skip_sample]/(max(labels)+1))]),
            label=labels[i]
        )

    # Plot the temporal path
    plt.plot(tsne[::skip_sample,0], tsne[::skip_sample,1])
    
    # Remove duplicates labels from Legend
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels, ids = np.unique(labels, return_index=True)
    #handles = [handles[i] for i in ids]
    #plt.legend(handles, labels, loc='best')
    
    plt.savefig(tsne_path, dpi=300)
    plt.clf()

def save_multiple_tsne(tsne, labels_pred, labels_gt, tsne_path, tsne_title="t-SNE Feature space"):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle(tsne_title)
    cmap = plt.cm.get_cmap('hsv')

    # There are too many features, plot one every skip_sample
    skip_sample = 4

    # Plot the temporal path
    axs[0].plot(tsne[::skip_sample,0], tsne[::skip_sample,1], linewidth=1, linestyle='-', color='lightgray', zorder=1)
    axs[1].plot(tsne[::skip_sample,0], tsne[::skip_sample,1], linewidth=1, linestyle='-', color='lightgray', zorder=1)
    
    # Plot each data point
    for i, feature in enumerate(tsne[::skip_sample]):
        axs[0].scatter(feature[0], feature[1],
            s = 8,
            c = np.array([cmap(labels_gt[i*skip_sample]/(max(labels_gt)+1))]),
            label=labels_gt[i],
            zorder=2
        )
        axs[1].scatter(feature[0], feature[1],
            s = 8,
            c = np.array([cmap(labels_pred[i*skip_sample]/(max(labels_pred)+1))]),
            label=labels_pred[i],
            zorder=2
        )

        # Add first and last frame
        if i == 0 or i == len(tsne[::skip_sample])-1:
            axs[0].annotate("0" if i==0 else "N", [feature[0], feature[1]], fontsize=15)
            axs[1].annotate("0" if i==0 else "N", [feature[0], feature[1]], fontsize=15)

    axs[0].set_title("Ground Truth Labels", fontsize='small')
    axs[1].set_title("Clustering Labels", fontsize='small')
    
    # Remove duplicates labels from Legend
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels, ids = np.unique(labels, return_index=True)
    #handles = [handles[i] for i in ids]
    #plt.legend(handles, labels, loc='best')
    
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)   
    axs[1].xaxis.set_visible(False)   
    axs[1].yaxis.set_visible(False)
    
    plt.savefig(tsne_path, dpi=300)
    plt.clf()
    plt.close(fig)


def save_segmentation_plot(gt_labels, eval_labels_lst, segmentation_path, features_types=None):
    if features_types == None:
        features_types = [""]*len(eval_labels_lst)

    assert(len(eval_labels_lst) == len(features_types))

    cmap = plt.cm.get_cmap('hsv')
    fig, axs = plt.subplots(len(eval_labels_lst)+1, figsize=(20, 3))
    
    # Plot the Ground Truth
    for i, label in enumerate(gt_labels):
        if label == -1:
            rect_gt = patches.Rectangle((i, 0), 1, 1, fill=True, color='whitesmoke')
        else:
            rect_gt = patches.Rectangle((i, 0), 1, 1, fill=True, color=cmap(gt_labels[i]/(max(gt_labels)+1)))
        axs[0].add_patch(rect_gt)
    
    axs[0].set_xlim([0, len(gt_labels)])
    
    # Plot the next eval labels
    for ax, eval_labels in enumerate(eval_labels_lst):
        for i, label in enumerate(eval_labels):
            if label == -1:
                rect_eval = patches.Rectangle((i, 0), 1, 1, fill=True, color='whitesmoke')
            else:
                rect_eval = patches.Rectangle((i, 0), 1, 1, fill=True, color=cmap(eval_labels[i]/(max(eval_labels)+1)))
            axs[ax+1].add_patch(rect_eval)

        axs[ax+1].set_xlim([0, len(eval_labels)])
    
    labels = ["GT"] + features_types
    
    for i, ax in enumerate(axs):
        ax.set_ylim([0, 1])
        
        ax.xaxis.set_visible(False)    
        ax.yaxis.set_visible(False)

        ax.text(-10, ax.get_ylim()[1]/2, labels[i], ha="right", va='center', fontsize=16, weight='bold')
    
    plt.savefig(segmentation_path, dpi=300)
    plt.clf()
    plt.cla()
    plt.close(fig)


def save_affinity_matrix_plot(affinity, affinity_path):
    plt.figure()
    plt.imshow(affinity, interpolation='none')
    plt.savefig(affinity_path, dpi=1000)
    plt.clf()
    plt.cla() 
    
    
def save_multiple_affinity_matrix(matrix_list, matrix_names, matrix_path):
    num_matrix = len(matrix_list)
    fig, axs = plt.subplots(1, num_matrix)

    for i, ax in enumerate(axs):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.set_title("{}".format(matrix_names[i]), fontsize=4)
        ax.matshow(matrix_list[i])

    plt.savefig(matrix_path, dpi=1000, bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close(fig)



def train_loss(loss, bestepoch, updates, loss_path):
    plt.figure()
    plt.plot(loss, color='blue', marker='o',mfc='blue' ,label='Loss') #plot the data
    plt.xticks(range(0,len(loss)+1, 1)) #set the tick frequency on x-axis
    if updates[:-1]!= len(loss):
        updates = updates[:-1]
    
    if len(updates)>1:
        for i in range(len(updates)):
            if i ==0:
                plt.plot(updates[i],loss[updates[i]], marker='o',mfc='red',label='Updates')
            else:
                plt.plot(updates[i],loss[updates[i]], marker='o',mfc='red')
    elif len(updates)==1:         
        plt.plot(updates, loss[updates[0]], marker='o',mfc='red',label='Update')
    
    
    plt.axvline(x=bestepoch, ls='dashed',color='black',label='Minimun Loss at epoch {}'.format( bestepoch))
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(loss_path, dpi=1000)
    plt.clf()
    plt.cla()