# MPoSM

This repository contains the official code for the paper:

[Masked Part-Of-Speech Model: Does Modeling Long Context Help Unsupervised POS-tagging?](https://arxiv.org/abs/2206.14969)

Xiang Zhou, Shiyue Zhang and Mohit Bansal

NAACL 2022

This repo is still *work-in-progress*, more code for analysis and ablation experiments, as well as pretrained checkpoints will come soon.


### Dependencies
The code is tested on Python 3.7.9 and Pytorch 1.6.0.

Other dependencies are listed in `requirements.txt` and can be installed by running `pip install -r requirements.txt`

This repository uses [wandb](https://github.com/wandb/client) for logging experiments. So before running your experiments, you need to log in you wandb account.


### Datasets
#### Natural Language Datasets
For the English 45-tag Penn Treebank dataset, please obtain the data from LDC. After obtaining the data, place them under the `data` folder in the same format as `data/sample.txt`

For the universal treebank v2.0 dataset, please download the data from (https://github.com/ryanmcd/uni-dep-tb) [https://github.com/ryanmcd/uni-dep-tb] and also put under `data/`. The slight difference between the data format between these two datasets are controlled by the `--ud_format` argument in the experiment scripts.

Then before running any experiments, run `bash rechunk_and_concat_data.sh [INPUT DATA]`. This will create a rechunked version of that data with the name `[INPUT DATA].withrechunk`.

#### Synthetic Datasets for Agreement Learning Experiments
The synthetic datasets used for agreement learning experiments are placed under `data/synthetic`. Please refer to Sec. 7.1 and Appendix H in the paper for detailed descriptions.


### Experiments

In the current repository, we provide three example scripts demonstrating how to run three different variants of the MPoSM model used in our work. For this experiment, you also need to download the pretrained embeddings from [here](https://drive.google.com/file/d/1LFi6ovD6nwMiEHvm26pklteTUUzcohDy/view?usp=sharing) and change the `--word_vec` argument in the script.

To run the English experiments using pretrained embeddings, run `bash scripts/run_em_with_emb.sh [SEED]`

To run the experiments on the universal treebank that involves a pretraining step, run `bash scripts/run_uni_pretrain.sh [LANGUAGE] [SEED]`

To run the experiments on the universal treebank that also uses an mBERT encoder, run `bash scripts/run_uni_mbert.sh [LANGUAGE] [SEED]`

The results of all these experiments can be checked through the wandb interface.


### Acknowledgement
The code in this repository is based on [https://github.com/jxhe/struct-learning-with-flow](https://github.com/jxhe/struct-learning-with-flow)


### Reference
```
@inproceedings{zhou2022mposm,
  title={Masked Part-Of-Speech Model: Does Modeling Long Context Help Unsupervised POS-tagging?},
  author={Xiang Zhou and Shiyue Zhang and Mohit Bansal},
  booktitle={The 2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2022}
}
```



