# Introduction

## About the Project

In this project, I have integrated the WideResnet model, pretrained on mini Imagenet with 100 classes, with an untrained DNN pipeline, with the architecture of 128, 64, 32, 16, bottleneck, 16, 32, 64, 128.

## To Download

### Dataset

1. Download the dataset from [my onedrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/satvik_verma_iiitb_ac_in/Eptf-_lau2FGiZ7E-WBJNd8BAaVoi_q-1gwji0Mfv9djSQ?e=PE33bj)

> Or you can download the dataset from the original source from [this link](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html).

> In this case copy can paste all image directories in a directory called ```all_images```.

### WideResNet Checkpoints

1. Download the WideResNet checkpoints from [my onedrive](https://iiitbac-my.sharepoint.com/:f:/g/personal/satvik_verma_iiitb_ac_in/Eptf-_lau2FGiZ7E-WBJNd8BAaVoi_q-1gwji0Mfv9djSQ?e=kLzFGJ).

2. Place the unzipped folder in the root of the project (where train.py is located).

## About the files

1. The WideResNet model is written in wrn_mixup_model.py.

2. The fully-connected DNN pipeline is in fully_connected.py.

3. From these files both are imported to train.py where, the CNN weights are loaded into the model, which are written in checkpoints_wideresnet/train_64_val_16_test_20/30.tar.

> Here I have used pretrained WideResNet that was trained on 64 train, 16 validation and 20 test split. Different model can alos be used. Update the train.py accordingly.

4. In order to change the train, validation and test proportions for the model, go to ``filelists/miniImagenet/make_json.py`` and change the train_split, val_split and test_split variables accordingly. Run the file.

5. To change batch_size, epoches, checkpoint frequency, alpha, bottleneck and mixup or no mixup make changes in ``run.sh`` file.


# How to run

1. Setup the conda environment through ``environment.yaml`` file, using the command

```

conda env create -f environment.yaml

```
2. Run file run.sh.
