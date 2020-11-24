#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/11/12 10:37 
# ide： PyCharm
import json
import os

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from train.utils import Evaluator, Data_Generator
from train.data_process import data_process
from bert4keras.backend import K, batch_gather
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import open
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
import numpy as np

model_root_path = rootPath + '/model/'
corpus_root_path = rootPath + '/corpus/'


class ReextractBertTrainHandler():
    def __init__(self, params, Train=False):
        self.bert_config_path = model_root_path + "chinese_L-12_H-768_A-12/bert_config.json"
        self.bert_checkpoint_path = model_root_path + "chinese_L-12_H-768_A-12/bert_model.ckpt"
        self.bert_vocab_path = model_root_path + "chinese_L-12_H-768_A-12/vocab.txt"
        self.tokenizer = Tokenizer(self.bert_vocab_path, do_lower_case=True)
        self.model_path = model_root_path + "best_model.weights"
        self.params_path = model_root_path + 'params.json'
        gpu_id = params.get("gpu_id", None)
        self._set_gpu_id(gpu_id)  # 设置训练的GPU_ID
        self.memory_fraction = params.get('memory_fraction')
        if Train:
            self.train_data_file_path = params.get('train_data_path')
            self.valid_data_file_path = params.get('valid_data_path')
            self.maxlen = params.get('maxlen', 128)
            self.batch_size = params.get('batch_size', 32)
            self.epoch = params.get('epoch')
            self.data_process()
        else:
            load_params = json.load(open(self.params_path, encoding='utf-8'))
            self.maxlen = load_params.get('maxlen')
            self.num_classes = load_params.get('num_classes')
            self.p2s_dict = load_params.get('p2s_dict')
            self.i2p_dict = load_params.get('i2p_dict')
            self.p2o_dict = load_params.get('p2o_dict')
        self.build_model()
        if not Train:
            self.load_model()

    def _set_gpu_id(self, gpu_id):
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def data_process(self):
        self.train_data, self.valid_data, self.p2s_dict, self.p2o_dict, self.i2p_dict, self.p2i_dict = data_process(
            self.train_data_file_path, self.valid_data_file_path, self.maxlen, self.params_path)
        self.num_classes = len(self.i2p_dict)
        self.train_generator = Data_Generator(self.train_data, self.batch_size, self.tokenizer, self.p2i_dict,
                                              self.maxlen)

    def extrac_subject(self, inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        subject_ids = K.cast(subject_ids, 'int32')
        start = batch_gather(output, subject_ids[:, :1])
        end = batch_gather(output, subject_ids[:, 1:])
        subject = K.concatenate([start, end], 2)
        return subject[:, 0]

    def build_model(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        if self.memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = self.memory_fraction
            config.gpu_options.allow_growth = False
        else:
            config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 补充输入
        subject_labels = Input(shape=(None, 2), name='Subject-Labels')
        subject_ids = Input(shape=(2,), name='Subject-Ids')
        object_labels = Input(shape=(None, self.num_classes, 2), name='Object-Labels')
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=self.bert_config_path,
            checkpoint_path=self.bert_checkpoint_path,
            return_keras_model=False,
        )
        # 预测subject
        output = Dense(units=2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(bert.model.output)
        subject_preds = Lambda(lambda x: x ** 2)(output)
        self.subject_model = Model(bert.model.inputs, subject_preds)
        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        output = bert.model.layers[-2].get_output_at(-1)
        subject = Lambda(self.extrac_subject)([output, subject_ids])
        output = LayerNormalization(conditional=True)([output, subject])
        output = Dense(units=self.num_classes * 2,
                       activation='sigmoid',
                       kernel_initializer=bert.initializer)(output)
        output = Lambda(lambda x: x ** 4)(output)
        object_preds = Reshape((-1, self.num_classes, 2))(output)
        self.object_model = Model(bert.model.inputs + [subject_ids], object_preds)
        # 训练模型
        self.train_model = Model(bert.model.inputs + [subject_labels, subject_ids, object_labels],
                                 [subject_preds, object_preds])
        mask = bert.model.get_layer('Embedding-Token').output_mask
        mask = K.cast(mask, K.floatx())
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        self.train_model.add_loss(subject_loss + object_loss)
        AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
        self.optimizer = AdamEMA(lr=1e-4)
        self.train_model.compile(optimizer=self.optimizer)

    def load_model(self):
        self.train_model.load_weights(self.model_path)

    def predict(self, text):
        """
        抽取输入text所包含的三元组
        text：str(<离开>是由张宇谱曲，演唱)
        """
        tokens = self.tokenizer.tokenize(text, max_length=self.maxlen)
        token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.maxlen)
        # 抽取subject
        subject_preds = self.subject_model.predict([[token_ids], [segment_ids]])
        start = np.where(subject_preds[0, :, 0] > 0.6)[0]
        end = np.where(subject_preds[0, :, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            spoes = []
            token_ids = np.repeat([token_ids], len(subjects), 0)
            segment_ids = np.repeat([segment_ids], len(subjects), 0)
            subjects = np.array(subjects)
            # 传入subject，抽取object和predicate
            object_preds = self.object_model.predict([token_ids, segment_ids, subjects])
            for subject, object_pred in zip(subjects, object_preds):
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append((subject, predicate1, (_start, _end)))
                            break
            return [
                (
                    [self.tokenizer.decode(token_ids[0, s[0]:s[1] + 1], tokens[s[0]:s[1] + 1]),
                     self.p2s_dict[self.i2p_dict[p]]],
                    self.i2p_dict[p],
                    [self.tokenizer.decode(token_ids[0, o[0]:o[1] + 1], tokens[o[0]:o[1] + 1]),
                     self.p2o_dict[self.i2p_dict[p]]],
                    (s[0], s[1] + 1),
                    (o[0], o[1] + 1)
                ) for s, p, o in spoes
            ]
        else:
            return []

    def train(self):
        evaluator = Evaluator(self.train_model, self.model_path, self.tokenizer, self.predict, self.optimizer,
                              self.valid_data)

        self.train_model.fit_generator(self.train_generator.forfit(),
                                       steps_per_epoch=len(self.train_generator),
                                       epochs=self.epoch,
                                       callbacks=[evaluator])
