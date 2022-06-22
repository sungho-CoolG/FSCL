# Fair Contrastive Learning
This paper is presented at **CVPR 2022**.
The implementation is based on the [source code](https://github.com/HobbitLong/SupContrast) of [supervised contrastive learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).  We are still clearing up it and will provide a more extended version including implementation of other comparable models. In this version, we cover implementation of  [SupCon](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) and [SimCLR](http://proceedings.mlr.press/v119/chen20j.html)  on CelebA and 


## Requirements

To install requirements:

```
conda env update -n env_name --file requirement.yaml
```



## Training and Evaluation

To train models (i.e., FSCL, FSCL+, FSCL*, SupCon,   SimCLR) in this paper on CelebA dataset, run this command:

**(1) Representation Learning**

Fair Supervised Contrastive Loss (FSCL)

```
python3 main_contrastive.py  --cosine --dataset celeba --method FSCL \
--group_norm 0 --name FSCL_CelebA --target_attribute_1 3 \
--sensitive_attribute_1 21 --data_folder ./datasets/CelebA/
```



Fair Supervised Contrastive Loss with Group-wise Normalization (FSCL+)

```
python3 main_contrastive.py  --cosine --dataset celeba --method FSCL \
--group_norm 1 --name FSCL_GroupNorm_CelebA --target_attribute_1 3 \
--sensitive_attribute_1 21 --data_folder ./datasets/CelebA/
```



Fair Supervised Contrastive Loss without Target Attribute Labels (FSCL*)

```
python3 main_contrastive.py  --cosine --dataset celeba --method FSCL* \
--name FSCL_WithoutTarget_CelebA --target_attribute_1 3 \
--sensitive_attribute_1 21 --data_folder ./datasets/CelebA/
```



Supervised Contrastive Loss (SupCon)

```
python3 main_contrastive.py  --cosine --dataset celeba --method SupCon \
--name SupCon_CelebA --target_attribute_1 3 \
--sensitive_attribute_1 21 --data_folder ./datasets/CelebA/
```



Self-supervised Contrastive Loss (SimCLR)

```
python3 main_contrastive.py  --cosine --dataset celeba --method SimCLR \
--name SimCLR_CelebA --target_attribute_1 3 \
--sensitive_attribute_1 21 --data_folder ./datasets/CelebA/
```

You might select if you apply the group-wise normalization by  `--group_norm ` .  Target  attributes (`--target_attribute_1 -- target_attribute_2`) and  sensitive attributes (`--sensitive_attribute_1 --sensitive_attribute_2`) are set by the number of attributes (e.g., 3: *attractiveness* and 21: *male*) in CelebA dataset or by the name of attributes (e.g., gender, age, ethnicity) in UTK Face dataset.



**(2) Classifier Training and Evaluation**  

```
python3 main_classifier.py --cosine --dataset celeba --target_attribute_1 3 --sensitive_attribute_1 21 --ckpt model.pth  --data_folder ./datasets/CelebA/
```



## Results
**Image Classification on CelebA ** (TA: *attractiveness* SA: *male*)

| Model  |   Supervision   | Accuracy(%) | Equalized Odds(%) |
| :----: | :-------------: | :---------: | :---------------: |
|  FSCL  |   Supervised    |    79.1     |       11.5        |
| FSCL+  |   Supervised    |    79.1     |        6.5        |
| SupCon |   Supervised    |    80.5     |       30.5        |
| FSCL*  | Semi-supervised |    74.6     |       14.8        |
| SimCLR | Self-supervised |    75.7     |       29.4        |



**Image Classification on UTK FACE ** (TA: *gender* SA: *ethnicity*)

Data imbalance ($\alpha$)=4

| Model  | Supervision | Accuracy(%) | Equalized Odds(%) |
| :----: | :---------: | :---------: | :---------------: |
|  FSCL  | Supervised  |    90.1     |        2.7        |
| FSCL+  | Supervised  |    90.1     |        1.6        |
| SupCon | Supervised  |    89.8     |       10.6        |

Data imbalance ($\alpha$)=3

| Model  | Supervision | Accuracy(%) | Equalized Odds(%) |
| :----: | :---------: | :---------: | :---------------: |
|  FSCL  | Supervised  |    92.3     |        1.7        |
| FSCL+  | Supervised  |    92.2     |        1.0        |
| SupCon | Supervised  |    91.6     |        8.4        |

Data imbalance ($\alpha$)=2

| Model  | Supervision | Accuracy(%) | Equalized Odds(%) |
| :----: | :---------: | :---------: | :---------------: |
|  FSCL  | Supervised  |    91.6     |        1.0        |
| FSCL+  | Supervised  |    91.5     |        0.6        |
| SupCon | Supervised  |    92.0     |        4.5        |



## Contributing

This repo is licensed under the terms of the MIT license.
