# SimCSE for TensorFlow v2
**中文** | [**English**](https://github.com/jifei/simcse-tf2)  

![Python Versions](https://img.shields.io/badge/Python-3.0+-blue.svg)
![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)

基于TensorFlow 2.x keras 开发的 SimCSE，支持无监督和有监督的训练.
## 例子
- 有监督. [代码](https://github.com/jifei/simcse-tf2/blob/master/examples/supervised_train.py)
- 增加负采样的有监督(包含随机负采样和难负采样). [代码](https://github.com/jifei/simcse-tf2/blob/master/examples/supervised_neg_train.py)
- 无监督. [代码](https://github.com/jifei/simcse-tf2/blob/master/examples/unsupervised_train.py)
- 阿里问天引擎训练部分. [代码](https://github.com/jifei/simcse-tf2/blob/master/examples/wentian_train.py)

[“阿里灵杰”问天引擎电商搜索算法赛](https://tianchi.aliyun.com/competition/entrance/531946/introduction?spm=5176.12281957.1004.5.38b02448HKvsCR) 数据处理，打包提交可以参考 [enze5088/WenTianSearch](https://github.com/enze5088/WenTianSearch) 中的代码。本代码中的训练结果可以达到0.24左右的成绩，有各种超参数可以自行选择和调试。

## 参考
- [bojone/Bert4Keras](https://github.com/bojone/bert4keras) & [bojone/SimCSE](https://github.com/bojone/SimCSE)
- [princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)
- [enze5088/WenTianSearch](https://github.com/enze5088/WenTianSearch)
- [muyuuuu/E-commerce-Search-Recall](https://github.com/muyuuuu/E-commerce-Search-Recall)