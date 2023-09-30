# DAFOSNET

This repo contains code accompanying the paper, 	[Domain Adaptive Few-Shot Open-Set
Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Pal_Domain_Adaptive_Few-Shot_Open-Set_Learning_ICCV_2023_paper.pdf). It includes code for running the few-shot domain adaptive semi-supervised learning experiments.

### Abstract

Few-shot learning has made impressive strides in addressing the crucial challenges of recognizing unknown samples from novel classes in target query sets and managing visual shifts between domains. However, existing techniques fall short when it comes to identifying target outliers under domain shifts by learning to reject pseudo-outliers from the source domain, resulting in an incomplete solution to both problems. To address these challenges comprehensively, we propose a novel approach called Domain Adaptive Few-Shot Open Set Recognition (DA-FSOS) and introduce a meta-learning-based architecture named DAFOSNET. During training, our model learns a shared and discriminative embedding space while creating a pseudo open-space decision boundary, given a fully-supervised source domain and a label-disjoint few-shot target domain. To enhance data density, we use a pair of conditional adversarial networks with tunable noise variances to augment both domains’ closed and pseudo-open spaces. Furthermore, we propose a domain-specific batch-normalized class prototypes alignment strategy to align both domains globally while ensuring class-discriminativeness through metric objectives. Our training approach ensures that DAFOS-NET can generalize well to new scenarios in the target domain. We present three benchmarks for DA-FSOS based on the Office-Home, mini-ImageNet/CUB, and DomainNet datasets and demonstrate the efficacy of DAFOSNet through extensive experimentation.

### Dependencies
This code requires the following:
* python 3.7+
* TensorFlow v2.12.0+
* keras_cv v1.4.0

### Data
The Domain Adaptive experiments are performed by choosing two domains from Office-Home, MiniImageNet-CUB, and DomainNet datasets.

Make two new folders, `data/raw` and `data/processed`

Keep the two domain folders of your choice from the aforementioned datasets in the `data/raw` folder.

For preprocessing details, take a look at `data/save_data_to_numpy.py`. By default, the flag for saving the input images tensor is set to "True" in `config.json` at `"pre_process":"True"`. In case that is not needed, make it "False". You don't have to run this script separately.

If the flag is "True" the numpy arrays will be saved at `data/processed`.

### Usage
Modify the settings in `config.json`

##### Checkpoints:
train checkpoints will be saved at `./checkpoints/train` and fintune checkpoints at `./checkpoints/finetune`

##### For training:
&nbsp;`python3 train.py`
##### For finetuning:
&nbsp;`python3 finetune.py`
##### For testing:
&nbsp;`python3 test.py`

## Citation  
If you use any content of this repo for your work, please cite the following bib entry:

	@InProceedings{Pal_2023_ICCV,
    author    = {Pal, Debabrata and More, Deeptej and Bhargav, Sai and Tamboli, Dipesh and Aggarwal, Vaneet and Banerjee, Biplab},
    title     = {Domain Adaptive Few-Shot Open-Set Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18831-18840}
	}

## Licence
DAFOS-NET is released under the MIT license.

Copyright (c) 2023 Debabrata Pal. All rights reserved.
