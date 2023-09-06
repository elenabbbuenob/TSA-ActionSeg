import glob
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FeaturesDataset(Dataset):
    def __init__(self, background, subset=False):
        super(FeaturesDataset, self).__init__()
        features_path = "/datasets/breakfast_action/features/"
        self.labels_path = "/datasets/breakfast_action/groundTruth/"
        self.features_files = sorted(glob.glob(features_path + "/**/*.txt"))
        self.mapping_file = '/datasets/breakfast_action/mappping/mapping.txt'
        
        if subset:
            self.features_files = [feature_file for feature_file in self.features_files if '_cam01_' in feature_file]
            
        
        self.background = background
	with open(self.mapping_file) as f:
            lines = f.readlines()
        
        self.map = dict()
        for map in lines:
            idx, action = map.replace('\n', '').split(' ')
            self.map[action] = idx
            
    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):

        features_file = self.features_files[idx]
        
        filename = os.path.basename(features_file).split('.')[0]
        labels_file = os.path.join(self.labels_path, filename)
        
        features = []
        labels_str = []
        labels = []

        with open(features_file) as f:
            lines = f.readlines()

        for feature in lines:
            features.append(np.array(feature.split(' ')).astype(float))

        with open(labels_file) as f:
            lines = f.readlines()

        for label in lines:
            label_str = str(label).replace('\n','')
            labels_str.append(label_str)
            labels.append(int(self.map[label_str]))

        video_len = len(labels_str)
        action_idx = None

        # Remove the background features --> one less label and one less cluster
        if self.background == False:
            action_idx = np.where(np.array(labels_str) != 'SIL')[0]
            labels_str = [labels_str[i] for i in action_idx]
            labels = [labels[i] for i in action_idx]
            features = [features[i] for i in action_idx]

        return {'features': torch.FloatTensor(features),
                'labels': labels,
                'labels_str': labels_str,
                'filename': filename,
                'action_idx': action_idx,
                'video_len': video_len }
