# PURE_ace05zh
[中文](README_zh.md)|[English](README.md)
pure name entity recognition model for ace2005 Chinese

a name entity recognition model base on [PURE](https://github.com/princeton-nlp/PURE) for Chinese ace2005 dataset. Comparing with origin PURE, I just removed allennlp package dependence.

step1: run `ace_parser.py` to parse ace2005 data, the code is based on [ace2005chinese_preprocess](https://github.com/ll0iecas/ace2005chinese_preprocess),I have modified the code to adapte PURE model.

step2: run `train.sh` to train model.

Result on [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) with 21 entity types including event-type:
```
INFO - root - P: 0.83622, R: 0.63672, F1: 0.72296
```
notice that PURE model evaluate the model useing a strict standard which means a correct result must have a correct start_index, end_index and entity_type, and the result is phrase level not tag level.