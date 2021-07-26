# triplet_extraction

三元组抽取（实体-关系-实体，实体-属性-属性值）-- 模型训练&预测

功能，可用于非结构化文本抽取三元组，在构建知识图谱三元组数据时可提供帮助作用

# 使用方式
## ·训练

​    调用train模块下的model_train.py文件

```shell
 python train/model_train.py
```



## ·预测

​	调用主工程目录下的model_predict.py文件
注意， 预测时优于json load 的原因，i2p_dict的键将变成str，但'p'类型是int。你需要将它的类型转换为str

```shell
 python train/model_train.py
```

# 数据及模型依赖

​	**·数据链接：https://pan.baidu.com/s/1D6JZMjQRonT83-MWgtsYsw** 

​	**·提取码：5tv3** 

​	**·链接：https://pan.baidu.com/s/1qoFBrw3rTPx4RFBwEYV7qA** 

​	**·提取码：ljxh** 

​	模型下载后放置于主工程的model目录下

​	数据下载后替换主工程的data目录

# 参考/感谢
实体关系联合抽取:https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py
