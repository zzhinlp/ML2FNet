# ML2FNet
Code for paper ML2FNet: A Simple but Effective Multi-Level Feature Fusion Network for Document-Level Relation Extraction


## Requirements
* allennlp==2.10.1
* fsspec==2023.5.0
* matplotlib==3.7.1
* numpy==1.24.3
* opt_einsum==3.3.0
* overrides==7.3.1
* pandas==2.0.1
* torch==1.12.1
* tqdm==4.65.0
* transformers==4.20.1
* ujson==5.7.0

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
ML2FNet
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

```bash
python train_docred.py # for DocRED
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

### CDR and GDA
Train CDA and GDA model with the following command:
```bash
python train_bio.py --dataset=CDR # for CDR
python train_bio.py --dataset=GDA # for GDA
```
