# -*- coding: utf-8 -*-
from decimal import Decimal
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
            #lh2=F.Linear(hidden_dim,hidden_dim),
            lo=F.Linear(hidden_dim, output_dim)
        )

        self.afi = self.__activation_func(input_af)
        self.afh = self.__activation_func(hidden_af)
        #self.afh2 = self.__activation_func(hidden_af)
        #if hidden2_af <> None:
         #   self.afh2 = self.__activation_func(hidden2_af)

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
        #u = self.lh2(z)
        #z = self.afh2(u)

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


def load_data(path, data_limit):
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
            if len(d) == 0 or re.compile(r'[\d"\']').match(d[0]) is None:
                continue

            if data_count > data_limit and data_limit > 0 :
                break
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
        if data_count > data_limit and data_limit > 0:
            break

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


KEY_STATISTICS = "statistics"
KEY_DATA_WAREHOUSE = "data_path"
SUF_TRAINING_DATA = "/train_data"
SUF_LABEL_DATA = "/train_label"
SUF_E_DATA = "/eval_data"
SUF_E_LABEL = "/eval_label"
KEY_OUTPUT = "output"
KEY_HIDDEN_DIM = "hidden_dim"
KEY_OPTIM = "optim"
KEY_INPUT_AF = "input_af"
KEY_HIDDEN_AF = "hidden_af"
KEY_HIDDEN2_AF = "hidden2_af"
KEY_HIDDEN2_AF = "hidden2_af"
KEY_OUTPUT_AF = "output_af"
KEY_EPOCH = "epoch"
KEY_BATCH_SIZE = "batch_size"
KEY_DATA_LIMIT = "data_limit"


def load_description(ini_file):
    inifile = ConfigParser.SafeConfigParser()
    inifile.read(ini_file)
    desc = dict()
    desc[KEY_STATISTICS] = inifile.get("path", "statictis")
    desc[KEY_DATA_WAREHOUSE] = inifile.get("path", "data_path")
    desc[KEY_OUTPUT] = inifile.get("path", "output_file")
    desc[KEY_HIDDEN_DIM] = inifile.getint("model", "hidden")
    desc[KEY_OPTIM] = inifile.get("model", "optimizer")
    desc[KEY_INPUT_AF] = inifile.get("model", "activation_input")
    desc[KEY_HIDDEN_AF] = inifile.get("model", "activation_hidden")
    desc[KEY_HIDDEN2_AF] = None
    if inifile.has_option("model", "activation_hidden2"):
        desc[KEY_HIDDEN2_AF] = inifile.get("model", "activation_hidden2")
    desc[KEY_OUTPUT_AF] = inifile.get("model", "activation_output")
    desc[KEY_EPOCH] = inifile.getint("training", "epoch")
    desc[KEY_BATCH_SIZE] = inifile.getint("training", "batch_size")
    if inifile.has_option("training", "data_limit"):
        desc[KEY_DATA_LIMIT] = inifile.getint("training", "data_limit")
    return padd_default_value(desc)

def padd_default_value(description):
    if not "data_limit" in description:
        description["data_limit"] = 0
    return description


def training_nn(desc):
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")   #これも上からわたるようにしよう。
    e_data = None
    e_label = None


    if not os.path.exists(desc[KEY_STATISTICS]):
        os.mkdir(desc[KEY_STATISTICS])

    statistics_path = desc[KEY_STATISTICS] + "/" + time_stamp

    logfile = statistics_path + "/log.txt"

    os.mkdir(statistics_path )

    data = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_TRAINING_DATA, desc[KEY_DATA_LIMIT])
    label = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_LABEL_DATA, desc[KEY_DATA_LIMIT])

    my_nn = nn(len(data[0]),len(label[0]),desc[KEY_HIDDEN_DIM], desc[KEY_INPUT_AF], desc[KEY_HIDDEN_AF], desc[KEY_OUTPUT_AF], desc[KEY_HIDDEN2_AF])
    model = loss(my_nn)
    optimizer = get_optimizer(desc[KEY_OPTIM])
    optimizer.setup(model)
    train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(data, label),batch_size=desc[KEY_BATCH_SIZE], shuffle=True)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (desc[KEY_EPOCH], 'epoch'), out=".")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/loss','validation/main/accuracy']))
    trainer.run()

    plt.clf()
    plt.plot(model.total_loss)
    plt.savefig(statistics_path + "/loss.png")
    plt.clf()
    plt.plot(model.total_accuracy)
    plt.savefig(statistics_path + "/acc.png")
    chainer.serializers.save_npz(desc[KEY_OUTPUT], my_nn)

    if desc[KEY_DATA_WAREHOUSE] + SUF_E_DATA <> None:
        data = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_E_DATA, 0)
        label = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_E_LABEL, 0)
        predicts = [[] for x in range(len(label[0]))]
        labels = [[] for x in range(len(label[0]))]

        loss_val = 0
        for idx, (d, l) in enumerate(zip(data, label)):
            predict = my_nn(np.array([d]))
            for x in range(len(predict.data)):
                pass
            tee(Template("no.$no predict:$predict, actual:$actual, of data:$data").substitute(no=str(idx), predict=str(["{0:.3f}".format(x) for x in predict.data[0]]), actual=str(["{0:.3f}".format(x) for x in l]), data=str(["{0:.3f}".format(x) for x in d])),logfile)
            for pidx, pd in enumerate(predict.data):
                predicts[pidx].append(pd)
                labels[pidx].append(l[pidx])
                loss_val += float(l[pidx]) - pd
        for ci, (pd, lb) in enumerate(zip(predicts,labels)):
            plt.clf()
            plt.plot(pd, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "-")
            plt.plot(lb, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "--")
            plt.savefig(statistics_path + "/eval_" + str(ci) + ".png")
        avgstr = ",".join(["{0:.3f}".format(x) for x in loss_val / float(idx + 1)])
        accstr = ",".join(["{0:.3f}".format(x) for x in loss_val])
        tee(Template("---loss accumrate:$acc, average:$avg").substitute(acc=accstr, avg=avgstr), logfile)
        #
        """
        for ci, pd in enumerate(predicts):
            plt.plot(pd, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "-")
        for ci, lb in enumerate(labels):
            plt.plot(lb, MATPLOT_COLOR_LIST[ci % len (MATPLOT_COLOR_LIST)] + "--")
        avgstr = ",".join(["{0:.3f}".format(x) for x in loss_val / float(idx + 1)])
        accstr = ",".join(["{0:.3f}".format(x) for x in loss_val])
        tee(Template("---loss accumrate:$acc, average:$avg").substitute(acc=accstr, avg=avgstr), logfile)
        plt.savefig(statistics_path + "/eval.png")
        plt.show()
        """
ini = "./nn/gen_nn_05.ini"
training_nn(load_description(ini))
