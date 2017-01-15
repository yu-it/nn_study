import numpy as np
import chainer
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import iterators, optimizers
from chainer import Link, Chain, ChainList
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from data_treating.number_image import treatment


import loader
import re
import os
from datetime import datetime

class Consts:
    #
    """
    batch_size = 10
    epoch = 20
    dataset_name = ""
    filter = ".+\.reg.+csv$"
    nn_x_width = 100
    nn_x_height = 100
    nn_conv1_out = 20
    nn_conv2_out = 50
    nn_kernel_size = 5
    nn_l2_per_l1 = 0.5
    """
    batch_size = 10
    epoch = 5
    dataset_name = ""
    filter = ".+\.reg.+csv$"
    data_limit = -1
    nn_x_width = 100
    nn_x_height = 100
    nn_conv1_out = 20
    nn_conv2_out = 50
    nn_kernel_size = 5
    nn_l2_per_l1 = 0.5
    test_data="2016_12_05_08_03_02"


def get_window_size(width, height):
    return [int(width * 0.5), int(height * 0.5)]


def data2vector(data_text):
    raw_array = np.array(data_text.split(","), dtype=np.float32)
    return np.reshape(raw_array,[1,int(np.sqrt(len(raw_array))),int(np.sqrt(len(raw_array)))])


def label2vector(label_text):
    raw_array = label_text.split("|")
    width, height = [int(x) for x in raw_array[1:3]]
    l,t,r,b= [int(x) for x in raw_array[3:]]
    area_size_x,area_size_y = get_window_size(width, height)
    stride_x = int(area_size_x / 2)
    stride_y = int(area_size_y / 2)
    vector = np.zeros([9], dtype=np.int32)
    for x in range(3):
        for y in range(3):
            if stride_x * x < l and r < stride_x * (x + 1) and stride_y * y < t and b < stride_y * (y + 1):
                vector[y * 3 + x] += 1
    return vector




class cnn(Chain):

    def __init__(self):

        conv1_output_width = (Consts.nn_x_width - (Consts.nn_kernel_size - 1)) / 2
        conv1_output_height = (Consts.nn_x_height - (Consts.nn_kernel_size - 1)) / 2
        conv2_output_width = (conv1_output_width  - (Consts.nn_kernel_size - 1)) / 2
        conv2_output_height = (conv1_output_height - (Consts.nn_kernel_size - 1)) / 2
        l1_input_size = conv2_output_height * conv2_output_width * Consts.nn_conv2_out
        l2_input_size = int(l1_input_size * Consts.nn_l2_per_l1)

        super(cnn, self).__init__(
            conv1=F.Convolution2D(1, Consts.nn_conv1_out, Consts.nn_kernel_size),  # 入力1枚、出力20枚、フィルタサイズ5ピクセル
            conv2=F.Convolution2D(Consts.nn_conv1_out, Consts.nn_conv2_out, Consts.nn_kernel_size),  # 入力20枚、出力50枚、フィルタサイズ5ピクセル
            l1=F.Linear(l1_input_size, l2_input_size),
            l2=F.Linear(l2_input_size, 1))
    def __call__(self, x):
        conv1_out = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        conv2_out = F.max_pooling_2d(F.relu(self.conv2(conv1_out)), 2)
        u1 = self.l1(conv2_out)
        z1 = F.dropout(F.relu(u1), train=True)
        u2 = self.l2(z1)
        return F.sigmoid(u2)
        #return u2

class calcrator(Chain):
    def __init__(self, _predictor):
        super(calcrator, self).__init__(predictor=_predictor)

    def __call__(self, x, t):
         y = self.predictor(x)
         loss = F.sigmoid_cross_entropy(y, t)
         accuracy = F.accuracy(y, t)
         report({'loss': loss, 'accuracy': accuracy}, self)
         print({'loss': loss, 'accuracy': accuracy}, self)
         return loss

my_cnn = cnn()
model = L.Classifier(my_cnn,lossfun=F.mean_squared_error,accfun=F.mean_squared_error)
#model = L.Classifier(my_cnn)
optimizer = optimizers.SGD()
optimizer.setup(model)
datas, labels = loader.load_data_and_label("2016_12_04_21_00_00", Consts.filter, data2vector, label2vector, Consts.data_limit)
labels = np.reshape(labels , [9,len(labels),1])
labels =  np.array(labels[0], dtype=np.float32)
print("aaa")
train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(datas, labels),batch_size=10, shuffle=True)

edatas, elabels = loader.load_data_and_label(Consts.test_data, Consts.filter, data2vector, label2vector, Consts.data_limit)
elabels = np.reshape(elabels , [9,len(elabels),1])
elabels =  elabels[0]

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (Consts.epoch, 'epoch'), out='result')
#trainer.extend(extensions.Evaluator(eval_iter, model))
#trainer.extend(extensions.Evaluator(train_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())
trainer.run()

for idx, (data,label) in enumerate(zip(edatas,elabels)):
    probability = F.sigmoid(my_cnn(np.array([data])))
    print(label)
    print(probability.data)
    pass
