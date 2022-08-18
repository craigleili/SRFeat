# SRFeat: Learning Locally Accurate and Globally Consistent Non-Rigid Shape Correspondence
By Lei Li, Souhaib Attaiki, Maks Ovsjanikov. (3DV 2022)

In this work, we present a novel learning-based framework that combines the local accuracy of contrastive learning with the global consistency of geometric approaches, for robust non-rigid matching. We first observe that while contrastive learning can lead to powerful point-wise features, the learned correspondences commonly lack smoothness and consistency, owing to the purely combinatorial nature of the standard contrastive losses. To overcome this limitation we propose to boost contrastive feature learning with two types of smoothness regularization that inject geometric information into correspondence learning. With this novel combination in hand, the resulting features are both highly discriminative across individual points, and, at the same time, lead to robust and consistent correspondences, through simple proximity queries. Our framework is  general and is applicable to local feature learning in both the 3D and 2D domains. We demonstrate the superiority of our approach through extensive experiments on a wide range of challenging matching benchmarks, including 3D non-rigid shape correspondence and 2D image keypoint matching.

![teaser](res/teaser.png)

## Link

[Paper](http://www.lix.polytechnique.fr/~maks/papers/SRFeat_3DV22.pdf)

## Citation
```
@inproceedings{li2022srfeat,
  title={{SRFeat}: Learning Locally Accurate and Globally Consistent Non-Rigid Shape Correspondence},
  author={Li, Lei and Attaiki, Souhaib and Ovsjanikov, Maks},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2022},
  organization={IEEE}
}
```

## 3D Shape Matching

### Dependencies
- CUDA 11
- Python 3.8
- Pytorch 1.8.1

Other used Python packages are listed in `requirements.txt`.
It is preferable to create a new conda environment for installing the packages.

The docker image that has a complete running environment can be found [here](https://hub.docker.com/r/craigleili/srfeat/tags).


### Data

The data and pretrained models can be found [here](https://1drv.ms/u/s!Alg6Vpe53dEDgbdki2r-v_CmJ67-jw?e=FjIE2D).
Extract the content of the zipped file to the root directory of the code.


### Training \& Testing

```
python trainer_srfeatd.py skip_train=False config=exp/srfeatd_***.yml 

python trainer_srfeats.py skip_train=False config=exp/srfeats_***.yml
```

### Use Pretrained Model

```
python <<trainer_srfeatd.py/trainer_srfeats.py>> skip_train=True test_ckpt=<<exp/../ckpt_latest.pth>> path_prefix=. log_dir=exp/log data.data_root=exp/data
```

### Evaluation
First, compute geodesic distance matrices by running `scripts/computeGeoDistMat.m`.

Then,
```
python eval_corr.py --data_root exp/data --test_roots exp/log/<<FolderName>> ...
```


## Image Keypoint Matching

Please check the branch `dgmc`.

## References
1. Donati et al. [Deep Geometric Maps: Robust Feature Learning for Shape Correspondence](https://github.com/LIX-shape-analysis/GeomFmaps). CVPR 2020.
1. Xie et al. [PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding](https://github.com/facebookresearch/PointContrast). ECCV 2020.
1. Sharp et al. [DiffusionNet: Discretization Agnostic Learning on Surfaces](https://github.com/nmwsharp/diffusion-net). TOG 2022.
1. Fey et al. [Deep Graph Matching Consensus](https://github.com/rusty1s/deep-graph-matching-consensus). ICLR 2020.


[![Creative Commons License](https://i.creativecommons.org/l/by-nc/4.0/80x15.png)](http://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
