# TSA
The repository contains our code for the proposed method described in our paper [**A deep metric learning approach for action segmentation**](https://arxiv.org/abs/2304.06403) inline link.
  
-----
### CODE AVAILABLE SOON ✨
-----

## Abstract:
In this paper, we propose a novel fully unsupervised framework that learns action representations suitable for the action segmentation task from the single input video itself, without requiring any training data. Our method is a deep metric learning approach rooted in a shallow network with a triplet loss operating on similarity distributions and a novel triplet selection strategy that effectively models temporal and semantic priors to discover actions in the new representational space. Under these circumstances, we successfully recover temporal boundaries in the learned action representations with higher quality compared with existing unsupervised approaches. The proposed method is evaluated on two widely used benchmark datasets for the action segmentation task and it achieves competitive performance by applying a generic clustering algorithm on the learned representations.

## Training and evaluating the model
The train and evaluation of the model is carried together. The process follows for each video:
1. Load initial IDT featureas
2. Train a shallow neural network to predict TSA features
3. Evaluate the action segmentation by clustering the learned TSA features.

## Citation

····# @inproceedings{xu2022covos,
····  title={Accelerating Video Object Segmentation with Compressed Video},
····  author={Kai Xu and Angela Yao},
····  booktitle={CVPR},
····  year={2022}
···· }
