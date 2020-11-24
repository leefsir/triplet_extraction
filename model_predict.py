#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/11/24 14:42 
# ide： PyCharm
import os
import sys

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)
from train.entity_relation_extract import ReextractBertTrainHandler

params = {}
model = ReextractBertTrainHandler(params)

text = "马志舟，1907年出生，陕西三原人，汉族，中国共产党，任红四团第一连连长，1933年逝世"
print(model.predict(text))