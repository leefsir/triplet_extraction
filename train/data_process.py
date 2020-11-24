#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/11/12 10:55
# ide： PyCharm

import json


def data_process(train_data_file_path, valid_data_file_path, max_len, params_path):
    train_data = json.load(open(train_data_file_path, encoding='utf-8'))

    if valid_data_file_path:
        train_data_ret = train_data
        valid_data_ret = json.load(open(valid_data_file_path, encoding='utf-8'))
    else:
        split = int(len(train_data) * 0.8)
        train_data_ret, valid_data_ret = train_data[:split], train_data[split:]
    p2s_dict = {}
    p2o_dict = {}
    predicate = []

    for content in train_data:
        for spo in content.get('new_spo_list'):
            s_type = spo.get('s').get('type')
            p_key = spo.get('p').get('entity')
            o_type = spo.get('o').get('type')
            if p_key not in p2s_dict:
                p2s_dict[p_key] = s_type
            if p_key not in p2o_dict:
                p2o_dict[p_key] = o_type
            if p_key not in predicate:
                predicate.append(p_key)
    i2p_dict = {i: key for i, key in enumerate(predicate)}
    p2i_dict = {key: i for i, key in enumerate(predicate)}
    save_params = {}
    save_params['p2s_dict'] = p2s_dict
    save_params['i2p_dict'] = i2p_dict
    save_params['p2o_dict'] = p2o_dict
    save_params['maxlen'] = max_len
    save_params['num_classes'] = len(i2p_dict)
    # 数据保存
    json.dump(save_params,
              open(params_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    return train_data_ret, valid_data_ret, p2s_dict, p2o_dict, i2p_dict, p2i_dict
