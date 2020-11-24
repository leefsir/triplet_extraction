#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/11/12 15:17 
# ide： PyCharm
import os
import sys

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)
from train.entity_relation_extract import ReextractBertTrainHandler

params = {
    "maxlen": 128,
    "batch_size": 32,
    "epoch": 1,
    "train_data_path":rootPath + "/data/train_data.json",
    "dev_data_path": rootPath + "/data/valid_data.json",
}

model = ReextractBertTrainHandler(params, Train=True)

model.train()
text = "马志舟，1907年出生，陕西三原人，汉族，中国共产党，任红四团第一连连长，1933年逝世"
print(model.predict(text))
