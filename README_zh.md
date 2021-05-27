# PURE_ace05zh
[中文](README_zh.md)|[English](README.md)
[PURE](https://github.com/princeton-nlp/PURE)模型命名实体识别的中文ace2005修改版，移除掉了allennlp包依赖，添加了中文ace2005的数据处理代码，代码基于[ace2005chinese_preprocess](https://github.com/ll0iecas/ace2005chinese_preprocess)，做了些许改动让解析后的数据符合`PURE`模型

1. 运行`ace_parser.py`
2. 运行`train.sh`
3. 在 [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) 上的结果，总共21个标签包括事件触发词类型
```
INFO - root - P: 0.83622, R: 0.63672, F1: 0.72296
```
`PURE`模型的评价标注比较严格，一个实体词的开头结尾及类型全部正确才算是一个正确的推断，最终的结果是词级别的而不是标签级别的