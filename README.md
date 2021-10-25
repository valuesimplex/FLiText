# FLiText:  A Faster and Lighter Semi-Supervised Text Classification with Convolution Networks
## Overview
[FLiText](https://arxiv.org/abs/2110.11869) is a semi-supervised distillation method which achieves state-of-the-art results on a wide variety of language.

Compared with SOTA semi-supervised learning methods, FLiText improves the accuracy on ligthweight model TextCNN from 50.00% to 90.49% on IMDb and from 39.8% to 58.06% on Yelp-5 and from 55.3% to 65.08% on Yahoo! Answer.

Compared with the fully supervised method, using less than 10% of labeled data, the performance is improved by 6.28%, 4.08%, and 3.81% on the datasets of IMDb, Yelp-5 and Yahoo! Answer respectively.


## Getting Started
### Requirement
- torch==1.6.0
- pandas==1.1.1
- numpy==1.18.5
- tensorflow==1.14.0
- keras-bert==0.84.0

### Download the BERT and GLOVE
Please download the BERT and unzip it to current dir. You can find [BERT](https://github.com/google-research/bert) and [GLOVE](https://drive.google.com/file/d/1avbyGPuuVRs5XLm5aQcQBtm7Em1pkyAW/view?usp=sharing)

### Download the data and process
First, Download the dataset and put them in "data" folder. You can find [IMDb](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), [Yelp-5](https://www.kaggle.com/yelp-dataset/yelp-dataset), [Yahoo! Answer](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset).

Second, process your data into the following format:

for labeled or test dataset, each line has a sentence and a label, separated by "tab"
for unlabeled dataset, each line has a ori sentence, a back translated sentence and a label identifier
```
### sup/test data format
sentence \t label \n

### unsup data format
ori_sentence \t label \t aug_sentence \n
```
Finally, please run `sh ./script/Process.sh`, it will process the data into the format required by the model

For IMDb, the original dataset has 25,000 labeled data, 50,000 unlabeled data and 25,000 test data. We randomly sample 20, 500 and
25,00 sentences as labeled data. Delete the labels of the remaining labeled data and mix with the original unlabeled data, 
the number of unlabeled data is 70000.

For Yelp-5 and Yahoo, we randomly sample 70,000 sentences of unlabeled data, and 5,000 sentences as test data to verify the SSL
method, and 500, 1000, and 2500 sentences as labeled data.

We use the Google Translate API to construct back translation data.

### Dir Structure
When you have finished the above steps, your dir structure should look like:
```
-data/
    -process.py              --> Code for data process
    -your labeled file       --> labeled data
    -your unlabeled file     --> unlabled data
    -your test file          --> test data
-inspirer/                   --> Codes for training inspirer network
-script/                     --> script file
-Target/                     --> Codes for training target network
-uncased_L-12_H-768_A-12/    --> Pre-trained BERT model
```

## Training
We use some example to show how to train FLiText step by step.

### Training Inspirer Network
Please run `sh ./script/Inspirer.sh` to train the Inspirer Network, you can modify the hyperparameters in `Inspirer/config
/yahoo-500.json` for example:
```
{
    "seed": 42,
    "lr": 2e-5,
    "warmup": 0.1,
    "do_lower_case": true,
    "mode": "train_eval",
    "uda_mode": true,

    "total_steps": 10000,
    "max_seq_length": 256,
    "train_batch_size": 8,
    "eval_batch_size": 16,

    "unsup_ratio": 3,
    "uda_coeff": 1,
    "tsa": "linear_schedule",
    "uda_softmax_temp": -1,
    "uda_confidence_thresh": -1,

    "data_parallel": true,
    "need_prepro": false,
    "sup_data_dir": "Inspirer/data/yahoo/yahoo_sup_train_500.txt",
    "unsup_data_dir": "Inspirer/data/yahoo/yahoo_unsup_train.txt",
    "eval_data_dir": "Inspirer/data/yahoo/yahoo_sup_test.txt",

    "model_file":null,
    "pretrain_file": "../uncased_L-12_H-768_A-12/bert_model.ckpt",
    "vocab":"../uncased_L-12_H-768_A-12/vocab.txt",
    "task": "yahoo",

    "save_steps": 100,
    "check_steps": 250,
    "results_dir": "results/yahoo",

    "is_position": false
}
```
After the training, the Inspirer model file will save to `Inspirer/results/yahoo/save/model_500_sup.pt`.

### Training Target Network
Please run `./script/Target.sh` to train the Target Network, you can modify the hyperparameters in `Target/config/yahoo/yahoo-500.json`

After the training, the Target model file will save to `Target/results/yahoo/save/model_500_sup.pt`.

