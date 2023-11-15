import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
import math
from sklearn.preprocessing import MinMaxScaler 

from model.mlp import MLP
from losses.losses import kl_loss
from utils.config import create_config
from utils.train import adjust_learning_rate, train
from utils.config import mkdir_if_missing
from utils.evaluate_utils import cluster_features, get_hungarian_matching, hungarian_evaluate, levenstein_distance, compute_f1_score, compute_selfsimilarity
from utils.plot_utils import save_segmentation_plot, save_multiple_affinity_matrix, compute_tsne, save_multiple_tsne, train_loss
from utils.config import mkdir_if_missing
from utils.compute_similarity import  similaridad_norm 

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Neural network')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--gpu', help='GPU device to use')
parser.add_argument('--name', help='Experiment name')
args = parser.parse_args()

# Set the same seed always:
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

# Check if GPU is available
if args.gpu is not None:
    cuda = torch.cuda.is_available()
else:
    cuda = False

# Set GPU device if available
if cuda:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.set_device(device)

# Metrics evaluation
cluster_techniques = ['kmeans', 'finch', 'spectral', 'twfinch']
IOU = [0]*len(cluster_techniques)
MOF = [0]*len(cluster_techniques)
EDIT = [0]*len(cluster_techniques)
F1 = [0]*len(cluster_techniques)


# Retrieve the config file
p = create_config(args.config_exp)
print("\n".join("{}\t{}".format(k, v) for k, v in p.items()))

# Depending on the dataset name, import the appropriate FeaturesDataset
if p['db_name'] == 'breakfast_action':
    from data.breakfast_action import FeaturesDataset
    features_dataset = FeaturesDataset(p['consider_background'], p['subset'])
    avg_actions = np.array([4, 4, 4, 4, 5, 5, 5, 6, 7, 9])
    Nc = 5 
elif p['db_name'] == 'inria_yt':
    from data.inria_yt import FeaturesDataset
    features_dataset = FeaturesDataset(p['consider_background'])
    avg_actions = np.array([8, 6, 10, 10, 7])
    Nc= 9 
else:
    print("Invalid dataset name")
    sys.exit()
    
# Convert features_dataset to numpy array for indexing
features_dataset_np = np.array(features_dataset)

# Iterate through each video sample in the dataset
num_videos = 0 
for video_idx, sample in enumerate(features_dataset_np):
    # Plotting variables
    similarity_matrix_lst = []
    similarity_matrix_titles = [] 
    
    # Set the same seed always:
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    # Load the GT labels and file name
    labels_gt = sample['labels']        
    filename = sample['filename']
    action_idx = sample['action_idx']
    print("{}/{} \t {}".format(video_idx, len(features_dataset_np), filename))
    
    # Define the output path based on experiment name
    if args.name is None:
        output_path = os.path.join(p['base_dir'], filename)
    else:
        output_path = os.path.join(p['base_dir'] + '_' + str(args.name), filename)
    print(output_path)

    # Check if TSA features are already computed for this video
    if os.path.exists(output_path):
        # Load metrics from file and update the averages
        for i, cluster_technique in enumerate(cluster_techniques):
            metrics_file = os.path.join(output_path, filename + ".metrics_{}.txt".format(cluster_technique))

            with open(metrics_file) as f:
                lines = f.readlines()

            iou = float(lines[0].split(' ')[1])
            mof = float(lines[1].split(' ')[1])
            edit = float(lines[3].split(' ')[1])
            f1 = float(lines[2].split(' ')[1])

            IOU[i] = (num_videos * IOU[i] + iou) / (num_videos + 1)
            MOF[i] = (num_videos * MOF[i] + mof) / (num_videos + 1)
            F1[i] = (num_videos * F1[i] + f1) / (num_videos + 1)
            EDIT[i] = (num_videos * EDIT[i] + edit) / (num_videos + 1)

            print("Avg: \t MoF: {:.2f} \t IoU: {:.2f} \t F1-score: {:.2f} \t Edit: {:.2f} \t ({})".format(
                MOF[i] * 100, IOU[i] * 100, F1[i] * 100, EDIT[i] * 100, cluster_technique))
        
        # Move to the next video
        num_videos += 1
        continue

    # Load normalized features, neighbors, and ground truth labels
    if cuda:
        features = F.normalize(sample['features'].cuda(), dim=1)
    else:
        features = F.normalize(sample['features'], dim=1)

    # Determine batch size or downsampling strategy
    if p['batch_size'] == 0 or p['batch_size'] > len(features) or len(features) < 1000:
        batch_size = len(features)
    else:
        batch_size = p['batch_size']
    ndownsampling = batch_size / len(features)
    positive_window = len(features) // Nc

    # Calculate semantic similarity
    semantic = similaridad_norm(features.cuda(), param=0.01)
    if cuda:
        semantic = torch.Tensor(semantic).cuda()
    else:
        semantic = torch.Tensor(semantic)
    semantic = F.normalize(semantic, dim=1)

    # Calculate temporal similarity
    beta = -2 * math.log(0.5) / positive_window
    if action_idx is None:
        temp_x = list(range(len(features)))
    else:
        temp_x = action_idx
    temporal = -1 + 2 * torch.exp(-beta * torch.abs(torch.transpose(torch.tensor([temp_x]), 1, 0) - torch.tensor([temp_x] * len(temp_x))))
    temporal = MinMaxScaler().fit_transform(temporal)
    temporal = torch.Tensor(temporal).cuda()

    print('Len total:', len(features), ' Downsampling:', ndownsampling, ' Batch', batch_size, ' real len', sample['video_len'])
    print('Positive window: ', positive_window)
    
    ############################################
    # Initialize the model, criterion, and optimizer
    model = MLP(ninput=features.shape[1], noutput=p['output_dim'], nfeat=p['hidden_dim'], nhidden=p['hidden_layers'],
                mt=temporal.cuda(), ms=semantic.cuda(), mtsinput=features.shape[0], ndownsampling=batch_size)
    criterion = kl_loss(p)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])

    # Move model, features, and criterion to GPU if available
    if cuda:
        model = model.cuda()
        features = features.cuda()
        criterion.cuda()

    # Store original affinity matrix
    init_features_similarity = compute_selfsimilarity(features.cpu().numpy())
    similarity_matrix_lst.append(init_features_similarity)
    similarity_matrix_titles.append("IDT")

    current_loss = 0
    best_epoch = 0
    lossfinal = []
    epochs_updates = []
    update = [True, 0]

    # Training loop
    for epoch in range(0, p['num_epoch']):
        # Adjust learning rate for this epoch
        lr = adjust_learning_rate(p, optimizer, epoch)
        # Perform training for this epoch
        output_features, loss = train(features, model, criterion, optimizer, batch_size)
        
        # Store the loss for this epoch
        lossfinal.append(loss)
        
        # Print progress for every 5 epochs
        if epoch % 5 == 0:
            print("Epoch {}/{} \t Loss: {:.3f} \t LR: {:.5f}".format(epoch, p['num_epoch'], loss, lr))
        
        # Save the affinity matrix at specified epochs
        if (epoch in [0, 3] or epoch % 5 == 0) and p['affinity_matrix']:
            output_features_similarity = compute_selfsimilarity(output_features.detach().cpu().numpy())
            similarity_matrix_lst.append(output_features_similarity)
            similarity_matrix_titles.append("TSA\n(Epoch {})".format(epoch))
        
        # Early stopping
        if epoch > 2 and len(lossfinal) > 2 and update[0] != True and abs(loss - lossfinal[-2]) < p['dist']:
            print('Stop criteria in ', epoch + 1)
            update = [True, update[1] + 1]
            epochs_updates.append(epoch + 1)
        else:
            update = [False, update[1]]
        
        # Save similarity matrix of best learned TSA features and exit
        if epoch == p['num_epoch'] - 1 or update[1] > 2:
            tsa_features = output_features
            tsa_features_similarity = compute_selfsimilarity(tsa_features.detach().cpu().numpy())
            similarity_matrix_lst.append(tsa_features_similarity)
            similarity_matrix_titles.append("TSA")
            print("Minimum loss at epoch {}".format(best_epoch))
            break
        
    ############################################    
    # Additional processing and analysis after training
    # Convert features to numpy arrays
    idt_features = features.detach().cpu().numpy()
    tsa_features = tsa_features.detach().cpu().numpy()
    
    # Create output directory if it doesn't exist
    mkdir_if_missing(output_path)
    
    # Plot training loss if required
    if p['loss']:
        train_loss(lossfinal, best_epoch, epochs_updates, os.path.join(output_path, '{}.loss.png'.format(filename)))
    
    # Save affinity matrix plots if required
    if p['affinity_matrix']:
        save_multiple_affinity_matrix(similarity_matrix_lst, similarity_matrix_titles,
                                      os.path.join(output_path, '{}.affinity.png'.format(filename)))

    similarity_matrix_lst = []
    similarity_matrix_titles = []
    
    # Save learned features and labels
    np.save(os.path.join(output_path, '{}.npy'.format(filename)), tsa_features)
    np.save(os.path.join(output_path, '{}.labels.npy'.format(filename)), labels_gt)
    
    # Perform clustering, evaluation, and plotting for each cluster technique
    for i, cluster_technique in enumerate(cluster_techniques):
        cluster_labels_tsa = cluster_features(tsa_features, len(set(labels_gt)), cluster_type=cluster_technique)
        cluster_labels_idt = cluster_features(idt_features, len(set(labels_gt)), cluster_type=cluster_technique)
        
        # Perform Hungarian matching and evaluation metrics
        categorical_labels_tsa, match_tsa = get_hungarian_matching(cluster_labels_tsa,  np.array(labels_gt), len(set(labels_gt)))
        categorical_labels_idt, match_idt = get_hungarian_matching(cluster_labels_idt,  np.array(labels_gt), len(set(labels_gt)))
        
        # Plot segmentation if background is not considered
        if not p['consider_background']:
            categorical_labels_tsa_bg = [-1] * sample['video_len']
            categorical_labels_idt_bg = [-1] * sample['video_len']
            labels_gt_bg = [-1] * sample['video_len']
            
            for idx, action in enumerate(sample['action_idx']):
                categorical_labels_tsa_bg[action] = categorical_labels_tsa[idx]
                categorical_labels_idt_bg[action] = categorical_labels_idt[idx]
                labels_gt_bg[action] = labels_gt[idx]
            
            save_segmentation_plot(labels_gt_bg, [categorical_labels_idt_bg, categorical_labels_tsa_bg], 
                                   os.path.join(output_path, '{}.segmentation_{}.png'.format(filename, cluster_technique)), features_types=["IDT", "TSA"])
        else:
            save_segmentation_plot(labels_gt, [categorical_labels_idt, categorical_labels_tsa], 
                                   os.path.join(output_path, '{}.segmentation_{}.png'.format(filename, cluster_technique)), features_types=["IDT", "TSA"])
        
        # Plot t-SNE plots if required
        if p['tsne_plot']:
            init_features_tsne = compute_tsne(idt_features)
            tsa_features_tsne = compute_tsne(tsa_features)
            save_multiple_tsne(tsa_features_tsne, categorical_labels_tsa, labels_gt,
                               tsne_path=os.path.join(output_path, '{}.tsne_tsa_{}.png'.format(filename, cluster_technique)), tsne_title="TSA Features")
            save_multiple_tsne(init_features_tsne, categorical_labels_idt, labels_gt,
                               tsne_path=os.path.join(output_path, '{}.tsne_idt_{}.png'.format(filename, cluster_technique)), tsne_title="IDT Features")
        
        # Evaluate and save metrics
        mof, iou, nmi, ari = hungarian_evaluate(categorical_labels_tsa, labels_gt, match_tsa)
        edit = levenstein_distance(categorical_labels_tsa, labels_gt, norm=True)
        f1 = compute_f1_score(categorical_labels_tsa, labels_gt)
        
        metrics = open(os.path.join(output_path, '{}.metrics_{}.txt'.format(filename, cluster_technique)), 'w')
        metrics.writelines("IoU: {:.4f}\n".format(iou))
        metrics.writelines("MoF: {:.4f}\n".format(mof))
        metrics.writelines("F1: {:.4f}\n".format(f1))
        metrics.writelines("Edit: {:.4f}\n".format(edit))
        metrics.close()
        
        # Update average metrics for this cluster technique
        IOU[i] = (num_videos * IOU[i] + iou) / (num_videos + 1)
        MOF[i] = (num_videos * MOF[i] + mof) / (num_videos + 1)
        F1[i] = (num_videos * F1[i] + f1) / (num_videos + 1)
        EDIT[i] = (num_videos * EDIT[i] + edit) / (num_videos + 1)
        
        print("Avg: \t MoF: {:.2f} \t IoU: {:.2f} \t F1-score: {:.2f} \t Edit: {:.2f} \t ({})".format(
            MOF[i] * 100, IOU[i] * 100, F1[i] * 100, EDIT[i] * 100, cluster_technique))
    
    # Update the number of videos processed
    num_videos += 1

# Save the final metrics
for i, cluster_technique in enumerate(cluster_techniques):
    path_metrics = 'metrics_' + cluster_technique + '.txt'
    metrics = open(os.path.join(p['base_dir'], path_metrics), 'w')
    
    print("FINAL METRICS ({})".format(cluster_technique))
    print("MoF: {:.4f}".format(MOF[i]))
    print("IoU: {:.4f}".format(IOU[i]))
    print("F1: {:.4f}".format(F1[i]))
    print("Edit: {:.4f}".format(EDIT[i]))
    
    metrics.writelines(cluster_technique + "\n")
    metrics.writelines("MoF: {:.4f}\n".format(MOF[i]))
    metrics.writelines("IoU: {:.4f}\n".format(IOU[i]))
    metrics.writelines("F1: {:.4f}\n".format(F1[i]))
    metrics.writelines("Edit: {:.4f}\n".format(EDIT[i]))

metrics.close()
