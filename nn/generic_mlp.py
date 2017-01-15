# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from chainer import reporter
import chainer
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import iterators, optimizers
from chainer import Link, Chain, ChainList
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import math
import data_treating.tone_data.treatment as treatment
import data_treating.tone_data.test_data as test_data
import re

from datetime import datetime
import ConfigParser
from string import  Template
import re
import os

EXT_OF_FLOAT = re.compile(r".+\.float$")
EXT_OF_INT = re.compile(r".+\.int$")

MATPLOT_COLOR_LIST = ["b","g","r","c","m","y"]

class nn(Chain):

    def __activation_func(self, name):
        if name ==  "id":
            return F.identity
        elif name == "relu":
            return F.relu
        elif name == "sigmoid":
            return F.sigmoid
        elif name == "tanh":
            return F.tanh
        else:
            return F.identity
    def __init__(self,input_dim, output_dim, hidden_dim, input_af, hidden_af, output_af, hidden2_af = None):
        hidden_dim = hidden_dim
        self.hidden_layer_count = hidden_dim
        super(nn, self).__init__(
            li=F.Linear(input_dim, hidden_dim),
            lh=F.Linear(hidden_dim,hidden_dim),
            lh2=F.Linear(hidden_dim,hidden_dim),
            lo=F.Linear(hidden_dim, output_dim)
        )

        self.afi = self.__activation_func(input_af)
        self.afh = self.__activation_func(hidden_af)
        self.afh2 = self.__activation_func(hidden_af)
        if hidden2_af <> None:
            self.afh2 = self.__activation_func(hidden2_af)

        self.afo = self.__activation_func(output_af)

    def __call__(self, x):

        #input layer
        u = self.li(x)
        #z1 = F.dropout(F.sigmoid(u1), train=True)
        z = self.afi(u)

        #hidden layer
        u = self.lh(z)
        z = self.afh(u)

        #hidden layer2
        u = self.lh2(z)
        z = self.afh2(u)

        #output layer
        u= self.lo(z)
        return self.afo(u)


class loss(Chain):
    compute_accuracy = True

    def __init__(self, predictor):
        super(loss, self).__init__(predictor=predictor)
        self.y = None
        self.loss = None
        self.accuracy = None
        self.total_loss = []
        self.total_accuracy = []
    def __call__(self, x, t):

        self.y = self.predictor(x)
        self.loss = F.mean_squared_error(self.y, t)
        self.total_loss.append(self.loss.data)
        reporter.report({'loss': self.loss}, self)
        #print('loss:' + str(self.loss.data))
        if self.compute_accuracy:
            self.accuracy = self.y
            self.total_accuracy.append(self.accuracy.data.sum())
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def load_data(path):
    ret = []
    if EXT_OF_FLOAT.match(path) <> None:
        type = np.float32
    elif EXT_OF_INT.match(path) <> None:
        type = np.int32
    else:
        type = np.float32
    data_length = 0
    data_count = 0
    for f in sorted(os.listdir(path)):
        with open(path + "/" + f) as r:
            ds = r.readlines()
        for d in ds:
            data_count += 1
            text_array = d.split(",")
            if data_length == 0:
                data_length = len(text_array)
            elif data_length <> len(text_array):
                raise Exception(Template("データ次数が異なるデータがあります。問題のデータ：$datafile, 次数：$dim, 他のデータの次数:$other_dim").substitute(datafile=f, dim=len(text_array),other_dim=data_length,))

            for i in range(len(text_array)):
                if text_array[i].strip() == "0":
                    text_array[i] = "0.01"
            ret.append(text_array)
    return np.array(ret, dtype= type).reshape([data_count,data_length])

def get_optimizer(name):
    if name == "sgd":
        return optimizers.SGD()
    elif name == "adam":
        return optimizers.Adam()
    else:
        return optimizers.SGD()
def tee(str, file):
    print(str)
    with open(file, "a") as w:
        w.write((str + "\r\n"))

def training_nn(inifile_path, output=None):
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    e_data = None
    e_label = None

    inifile = ConfigParser.SafeConfigParser()
    inifile.read(inifile_path)

    statistics = inifile.get("path", "statictis")
    training_data = inifile.get("path", "training_data")
    label_data = inifile.get("path", "training_label")
    if inifile.has_option("path", "eval_data"):
        e_data = inifile.get("path", "eval_data")
        e_label = inifile.get("path", "eval_label")
    if output == None:
        output = inifile.get("path", "output_file")

    hidden_dim = inifile.getint("model", "hidden")
    optim = inifile.get("model", "optimizer")
    input_af = inifile.get("model", "activation_input")
    hidden_af = inifile.get("model", "activation_hidden")
    hidden2_af = None
    if inifile.has_option("model", "activation_hidden2"):
        hidden2_af = inifile.get("model", "activation_hidden2")
    output_af = inifile.get("model", "activation_output")
    epoch = inifile.getint("training", "epoch")
    batch_size = inifile.getint("training", "batch_size")


    if not os.path.exists(statistics):
        os.mkdir(statistics)

    statistics_path = statistics + "/" + time_stamp

    logfile = statistics_path + "/log.txt"

    os.mkdir(statistics_path )

    data = load_data(training_data)
    label = load_data(label_data )

    my_nn = nn(len(data[0]),len(label[0]),hidden_dim, input_af, hidden_af, output_af, hidden2_af)
    model = loss(my_nn)
    optimizer = get_optimizer(optim)
    optimizer.setup(model)
    train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(data, label),batch_size=batch_size, shuffle=True)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=".")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/loss','validation/main/accuracy']))
    trainer.run()

    plt.clf()
    plt.plot(model.total_loss)
    plt.savefig(statistics_path + "/loss.png")
    plt.clf()
    plt.plot(model.total_accuracy)
    plt.savefig(statistics_path + "/acc.png")
    chainer.serializers.save_npz(output, my_nn)

    if e_data <> None:
        data = load_data(e_data)
        label = load_data(e_label)
        predicts = [[] for x in range(len(label[0]))]
        labels = [[] for x in range(len(label[0]))]

        plt.clf()
        loss_val = 0
        for idx, (d, l) in enumerate(zip(data, label)):
            predict = my_nn(np.array([d]))
            for x in range(len(predict.data)):
                pass
            tee(Template("no.$no predict:$predict, actual:$actual, of data:$data").substitute(no=str(idx), predict=str(predict.data), actual=str(l), data=str(d)),logfile)
            for pidx, pd in enumerate(predict.data):
                predicts[pidx].append(pd)
                labels[pidx].append(l[pidx])
                loss_val += l[pidx] - pd
        for ci, pd in enumerate(predicts):
            plt.plot(pd, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "-")
        for ci, lb in enumerate(labels):
            plt.plot(lb, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "--")
        tee(Template("---loss accumrate:$acc, average:$avg").substitute(acc=loss_val, avg = loss_val / float(idx + 1)), logfile)
        plt.savefig(statistics_path + "/eval.png")
        plt.show()

ini = "./#local/gen_nn_03.ini"
training_nn(ini)
