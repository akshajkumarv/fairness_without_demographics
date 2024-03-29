## Prerequisites

- python 3.7.8
- h5py 3.2.1
- hdf5 1.10.6
- matplotlib 3.2.2
- numpy 1.20.1
- pandas 1.1.3
- pytorch 1.7.1
- pytorch-transformers 1.2.0
- torchvision 0.8.2
- tqdm 4.62.0

## Dataset and Code

### CelebA Dataset
CelebA dataset can be downloaded from this [link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). 

After downloading, store following files/folders in the `[root_dir]/celeba/celeba-dataset` directory:

- `img_align_celeba/`
- `list_attr_celeba.csv`
- `list_bbox_celeba.csv`
- `list_eval_partition.csv`
- `list_landmarks_align_celeba.csv`

## Biased Classifier
To train a biased classifier, run ```python train_biased_classifier.py --lr [learning rate] --weight_decay [weight decay]```.
Next, to tune the hyper-parameters of the biased classifier, we propose to maximize the Euclidean distance between the means (EDM) of correctly and incorrectly classified examples. To calculate EDM for all epochs trained with a specific [learning rate] and [weight decay], run ```python calculate_bias.py --lr [learning rate] --weight_decay [weight decay]```.

The model with highest EDM is selected as the final biased classifier to output noisy sensitive attributes.

## Unbiased Classifier
We train unbiased classifier using [JTT](https://proceedings.mlr.press/v139/liu21f.html) framework. While JTT uses ground-truth sensitive information to tune the hyper-parameters, we use the noisy sensitive attributes to tune the hyper-parameters of the unbiased classifier.

JTT opertates in two stages:
* Stage 1: Identify misclasified training examples at the end of few steps of standard training. To train stage 1 classifier, run ```python jtt_train_stage_1.py --lr [learning rate] --weight_decay [weight decay]```
* Stage 2: Upweight the misclassified examples from stage 1 and train an unbiased classifier by running ```python jtt_train_stage_2.py --stage_1_epoch [Stage 1 epoch] --upsample [Upsampling factor] --lr [learning rate] --weight_decay [weight decay]``` 

To improve worst-group accuracy of the unbiased model by tuning the hyper-parameters of JTT using noisy sensitive attributes, run ```python worst_group_accuracy.py``` 


