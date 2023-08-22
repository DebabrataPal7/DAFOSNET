# DAFOSNET

This repo contains code accompaning the paper, 	[PAPER NAME](https://arxiv.org/abs/URL). It includes code for running the few-shot domain adaptive semi-supervised learning domain experiments.

### Dependencies
This code requires the following:
* python 3.7+
* TensorFlow v2.12.0+
* keras_cv v1.4.0

### Data
The Domain Adaptive experiments are perfomed by choosing two domains from Office-Home, MiniImageNet-CUB and DomainNet datasets.

Make two new folders `data/raw` and `data/processed`

Keep the two domain folders of your choice from the beforementioned datasets into the `data/raw` folder.

For preprocessing details take a look at `data/save_data_to_numpy.py`. By default the flag for saving the input images tensor is set to "True" in `config.json` at `"pre_process":"True"`. In case that is not needed, make it "False". You don't have to run this script separately.

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

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/X-TRON404/DAFOSNET/issues).
