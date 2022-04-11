## 医学实体抽取_GlobalPointer_torch
该项目仅支持NER实体识别
+ 更加节约参数、高效的版本请移步[此仓库](https://github.com/xhw205/Efficient-GlobalPointer-torch)
+ 支持实体关系识别/SPO抽取的版本请移步[此仓库](https://github.com/xhw205/GPLinker_torch)
### 介绍

思想来自于苏神 [GlobalPointer](https://kexue.fm/archives/8373)，原始版本是基于keras实现的，模型结构实现参考[现有 pytorch 复现代码](https://github.com/gaohongkui/GlobalPointer_pytorch)【感谢!】，基于torch百分百复现苏神原始效果。

### 数据集

中文医学命名实体数据集 [点这里申请，很简单](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1)，共包含九类医学实体

### 环境

+ python 3.8.1

+ pytorch==1.8.1
+ transformer==4.9.2
+ tqdm
+ numpy

### 预训练模型

1、笔者比较喜欢用RoBerta系列 [RoBERTa-zh-Large-PyTorch](https://github.com/brightmart/roberta_zh)

2、点这里直接[goole drive](https://drive.google.com/file/d/1yK_P8VhWZtdgzaG0gJ3zUGOKWODitKXZ/view)下载

### 运行

注意把train/predict文件中的 bert_model_path 路径改为你自己的

#### train

```python
python train_CME.py
```

#### predict

```
python predict_CME.py
```

### 效果

苏神的模型效果还是不错的！

![image-20210914093108205.png](https://i.loli.net/2021/09/14/a1Zj7d4ik9CoePU.png)

