# Codes and Data for Down-stream Tasks
This folder contains codes and data for down-stream tasks.

## Usage
You should first unzip the `data.zip` in each folders.

You then just need to install the requirements and `cd` to the corresponding folders, then run `bash myTrain.sh`.


## Download Pre-trained Models
You can download the pre-trained models on the Google Drive: [BERT-our](https://drive.google.com/file/d/1_rsP1fXcVRx7FE-uiOq1Z4D5ZdoEE93n/view?usp=sharing), [ELECTRA-our](https://drive.google.com/file/d/1U93vySqyRgr-cHQ3bJ8k9YNiYlDD_jOO/view?usp=sharing).

## Change Pre-trained Model
You can change the pre-trained models in the `myTrain.sh` script in the corresponding folders for each downstream task, by commenting/uncommenting the `pretrain_name` argument. Note that when changing between BERT and ELECTRA, you should also change the `model_type/size/name` accordingly.
