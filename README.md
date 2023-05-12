# TSA
The repository contains our code for the proposed method described in our paper [**A deep metric learning approach for action segmentation**](https://arxiv.org/abs/2304.06403) inline link.
  
-----
### CODE AVAILABLE SOON ✨
-----
![Overview of the proposed TSA framework illustrated on a sample video of the Breakfast Dataset: network architecture transforming the initial features X into the learned features Z through a shallow network with a novel triplet selection strategy and a triplet loss based on similarity distributions.](/figures/frameworkdef.jpg)


## Abstract:
In this paper, we propose a novel fully unsupervised framework that learns action representations suitable for the action segmentation task from the single input video itself, without requiring any training data. Our method is a deep metric learning approach rooted in a shallow network with a triplet loss operating on similarity distributions and a novel triplet selection strategy that effectively models temporal and semantic priors to discover actions in the new representational space. Under these circumstances, we successfully recover temporal boundaries in the learned action representations with higher quality compared with existing unsupervised approaches. The proposed method is evaluated on two widely used benchmark datasets for the action segmentation task and it achieves competitive performance by applying a generic clustering algorithm on the learned representations.

## Training and evaluating the model
The train and evaluation of the model is carried together. The process follows for each video:
1. Load initial IDT featureas
2. Train a shallow neural network to predict TSA features
3. Evaluate the action segmentation by clustering the learned TSA features.

## Citation
```
@inproceedings{buenobenito2023tsa,
  title={Leveraging triplet loss for unsupervised action segmentation},
  author={E. Bueno-Benito, B. Tura and M. Dimiccoli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),  2nd Workshop on Learning with Limited Labelled Data in conjunction with CVPR 2023},
  year={2023}
}
```
