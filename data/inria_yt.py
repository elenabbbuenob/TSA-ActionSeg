import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video, video
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np


class FeaturesDataset(Dataset):
    def __init__(self, background):
        super(FeaturesDataset, self).__init__()
        features_path = "/home/elena/Datasets/inria_yt/features/"
        self.labels_path = "/home/elena/Datasets/inria_yt/groundTruth/"
        self.features_files = sorted(glob.glob(features_path + "/**/*.txt"))
        self.encoder = LabelEncoder()

        self.background = background
        self.pca_dim = 64
        self.pca = PCA(n_components=self.pca_dim)
        self.tau = 0.75

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):

        features_file = self.features_files[idx]
        filename = os.path.basename(features_file).split('.')[0]
        labels_file = os.path.join(self.labels_path, filename)

        features = []
        labels_str = []

        with open(features_file) as f:
            lines = f.readlines()

        for feature in lines:
            features.append(np.array(feature.split('  ')[1:]).astype(float))

        with open(labels_file) as f:
            lines = f.readlines()

        for label in lines:
            labels_str.append(str(label).replace('\n',''))

        video_len = len(labels_str)
        action_idx = None

        # Remove the background features --> one less label and one less cluster
        if self.background == False:
            action_idx = np.where(np.array(labels_str) != '-1')[0]
            background_idx = np.where(np.array(labels_str) == '-1')[0]
            
            # Get a random (1-tau) background frames and add them to the action features to evaluate
            background_idx_tau = np.random.choice(background_idx, int((1-self.tau)*len(background_idx)), replace=False)
            action_idx = sorted(np.hstack([action_idx, background_idx_tau]))

            labels_str = [labels_str[i] for i in action_idx]
            features = [features[i] for i in action_idx]

        # Reduce feature dimensionality
        if min(len(features), len(features[0])) < self.pca_dim:
            reduced_pca = PCA(n_components=min(len(features), len(features[0])))
            features = reduced_pca.fit_transform(features)
        else:
            features = self.pca.fit_transform(features)

        return {'features': torch.FloatTensor(features),
                'labels': self.encoder.fit_transform(labels_str),
                'labels_str': labels_str,
                'filename': filename,
                'action_idx': action_idx,
                'video_len': video_len}
