# TSA
The repository contains our code for the proposed method described in our paper [**Leveraging triplet loss for unsupervised action segmentation**](https://arxiv.org/abs/2304.06403) inline link.
  
![Overview of the proposed TSA framework illustrated on a sample video of the Breakfast Dataset: network architecture transforming the initial features X into the learned features Z through a shallow network with a novel triplet selection strategy and a triplet loss based on similarity distributions.](/figures/frameworkdef.jpg)
#### Overview of the proposed TSA framework illustrated on a sample video of the Breakfast Dataset: network architecture transforming the initial features X into the learned features Z through a shallow network with a novel triplet selection strategy and a triplet loss based on similarity distributions.

-----

## Abstract:
In this paper, we propose a novel fully unsupervised framework that learns action representations suitable for the action segmentation task from the single input video itself, without requiring any training data. Our method is a deep metric learning approach rooted in a shallow network with a triplet loss operating on similarity distributions and a novel triplet selection strategy that effectively models temporal and semantic priors to discover actions in the new representational space. Under these circumstances, we successfully recover temporal boundaries in the learned action representations with higher quality compared with existing unsupervised approaches. The proposed method is evaluated on two widely used benchmark datasets for the action segmentation task and it achieves competitive performance by applying a generic clustering algorithm on the learned representations.

## Training and evaluating the model
The train and evaluation of the model is carried together. The process follows for each video:
1. Load initial IDT featureas
2. Train a shallow neural network to predict TSA features
3. Evaluate the action segmentation by clustering the learned TSA features.

To run the experiments and learn TSA features, run the following command:

```bash
python tsa.py --config_exp configs/'desired_dataset'.yml --gpu 'X' --name 'experiment_name'
```

There are three arguments to consider:
* The dataset yml file.
* The GPU ID number (0,1,2,...). If not provided, the script will run on CPU.
* The name of the experiment. If not provided, the experiment will be named as the dataset.

The metric reported in the conosle log is the average metric `1/n` over the `n` tested videos.
```
n/N          'video_name'
Avg:     MoF: XX.XX      IoU: XX.XX      Edit: XX.XX     F1-score: XX.XX         (kmeans)
Avg:     MoF: XX.XX      IoU: XX.XX      Edit: XX.XX     F1-score: XX.XX         (finch)
```

## Code tree
In this section we give a clue of what does every folder in this file structure do: 
* `configs/`: The configuration file for each dataset to be trained and evaluated. See `.yml` files for specific parameters.
* `data/`: The dataset classes of each of the different tested datasets. They load IDT features and generate a FeaturesDataset object class.
* `datasets/`: Contains the Improve Dense Trajectory (IDT) features for each one of the datasets.
* `losses/`: Triplet loss implementation.
* `models/`: The definition of the used MLP network
* `output/`: Generated ouput of the training and evaluation 
* `utils/`: The majority of the algorithm training, evaluation and clustering occurs in the files in this folder.


## Plotting the results
Segmentation results plot will be present in each `output/` folder, as well as the metrics obtained for each video. Affinity matrix and T-SNE plot of the original and learned TSA features will also be saved if configured in the config `.yml` file.

When all the dataset has been evaluated, a general average metric file will be generated reporting the metrics for all the dataset.

## Citation
```
@InProceedings{Bueno-Benito_2023_CVPR,
    author    = {Bueno-Benito, Elena and Vecino, Biel Tura and Dimiccoli, Mariella},
    title     = {Leveraging Triplet Loss for Unsupervised Action Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4921-4929}
}
```
