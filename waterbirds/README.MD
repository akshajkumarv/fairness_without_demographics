## Dataset and Code

### Waterbirds Dataset
Please refer to this [repo](https://github.com/kohpangwei/group_DRO) to download the Waterbirds dataset. 

After downloading, store following files/folders in the `[root_dir]/waterbirds` directory:

- `waterbird_complete95_forest2water2/`

## Biased Classifier
To train a biased classifier, run ```python train_biased_classifier.py --lr [learning rate] --weight_decay [weight decay]```.
Next, to tune the hyper-parameters of the biased classifier, we propose to maximize the Euclidean distance between the means (EDM) of correctly and incorrectly classified examples. To calculate EDM for all epochs trained with a specific [learning rate] and [weight decay], run ```python calculate_bias.py --lr [learning rate] --weight_decay [weight decay]```.

The model with highest EDM is selected as the final biased classifier to output noisy sensitive attributes.

## Unbiased Classifier
We train unbiased classifier using [JTT](https://proceedings.mlr.press/v139/liu21f.html) framework. While JTT uses ground-truth sensitive information to tune the hyper-parameters, we use the noisy sensitive attributes to tune the hyper-parameters of the unbiased classifier.

JTT opertates in two stages:
* Stage 1: Identify misclasified training examples at the end of few steps of standard training. To train stage 1 classifier, run ```python jtt_train_stage_1.py --lr [learning rate] --weight_decay [weight decay]```
* Stage 2: Upweight the misclassified examples from stage 1 and train an unbiased classifier by running ```python jtt_train_stage_2.py --stage_1_epoch [Stage 1 epoch] --upsample [Upsampling factor] --lr [learning rate] --weight_decay [weight decay]``` 

To improve worst-group accuracy of the unbiased model by tuning the hyper-parameters of JTT using noisy sensitive attributes,run ```python worst_group_accuracy.py``` 



