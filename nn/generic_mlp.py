# -*- coding: utf-8 -*-
import random
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


class NnTrainStatictis:
    def __init__(self):
        self.losses = []
        self.accurates = []
    def add_statistics(self, loss, accurate, x, y, t):
        self.losses.append(loss)
        self.accurates.append(accurate)


    def save(self, output):
        plt.clf()
        plt.plot(self.losses)
        plt.savefig(output + "/loss.png")
        #plt.clf()
        #plt.plot(model.total_accuracy)
        #plt.savefig(statistics_path + "/acc.png")
        pass


class NnResult:
    def __init__(self):
        self.datas = []
        self.predicts = []
        self.actuals = []
        self.count = 0
    def add_result(self, data, predict, actual):
        self.datas.append(data)
        self.predicts.append(predict.data[0])
        self.actuals.append(actual)
        self.count += 1
        pass

    def to_string(self, idx=-1):
        if idx < 0:
            idx = self.count - 1
        accuracy = compute_vector_similarly(self.actuals[idx], self.predicts[idx])
        return Template("no.$no accuracy:$acc, predict:$predict, actual:$actual, of data:$data")\
            .substitute(acc=accuracy,
                    no=str(idx),
                    predict=":".join(["{0:.3f}".format(x) for x in self.predicts[idx]]),
                    actual=":".join(["{0:.3f}".format(x) for x in self.actuals[idx]]),
                    data=":".join(["{0:.3f}".format(x) for x in self.datas[idx]]))
        pass

    def compute_summary_data(self):
        mean_accuracy = np.mean([compute_vector_similarly(self.actuals[i], self.predicts[i]) for i in xrange(self.count)])
        loss_accumrate = np.sum([np.sqrt((self.actuals[i] - self.predicts[i]) ** 2).sum() for i in xrange(self.count)])
        return [mean_accuracy, loss_accumrate]

    def to_string_summary(self):
        mean_accuracy, loss_accumrate = self.compute_summary_data()
        return Template("---total performance:$per, loss accumrate:$acc, average:$avg")\
            .substitute(per=mean_accuracy,acc=loss_accumrate, avg=loss_accumrate / float(self.count))
    def save(self, output):
        tee("\r\n".join([self.to_string(i) for i in range(self.count)]), output + "/log.txt")
        tee("\r\n" + self.to_string_summary(), output + "/log.txt")
        pd = self.predicts
        lb = self.actuals
        pd = np.transpose(pd)
        lb = np.transpose(lb)
        for idx, (p, l) in enumerate(zip(pd, lb)):
            plt.clf()
            #plt.plot(p, MATPLOT_COLOR_LIST[idx % len (MATPLOT_COLOR_LIST)] + "-")
            #plt.plot(l, MATPLOT_COLOR_LIST[idx % len (MATPLOT_COLOR_LIST)] + "--")
            plt.plot(p, "b--")
            plt.plot(l, "r--")
            plt.savefig(output + "/eval_" + str(idx) + ".png")
            plt.clf()
            plt.plot([compute_scalar_similarly(l[i], p[i]) for i in range(self.count)])
            plt.plot(lb, "b-")
            plt.savefig(output + "/accuracy_" + str(idx) + ".png")

        plt.clf()
        plt.plot([compute_vector_similarly(self.actuals[i], self.predicts[i]) for i in range(self.count)])
        plt.plot(lb, MATPLOT_COLOR_LIST[0 % len(MATPLOT_COLOR_LIST)] + "--")
        plt.savefig(output + "/accuracy.png")


    def show_fig(self, output):
        pass


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

    def __create_init_param_dict(self, input_dim, output_dim, input_af, output_af, hiddens):
        input_af = None #input_afってのは基本あんまないよね
        nn_param = dict()
        af_param = dict()
        seq = []
        tmp_hiddens = hiddens[:]
        tmp_hiddens.insert(0,[input_dim, input_af])
        tmp_hiddens.append([output_dim, output_af])
        node_in_count = input_dim
        node_out_count = -1
        node_af =  None
        for idx, node in enumerate(hiddens):
            node_out_count = node[0]
            node_name = "l" + str(idx)
            seq.append(node_name)
            nn_param[node_name] = F.Linear(node_in_count, node_out_count)
            if not node_af is None:
                af_param[node_name] = node_af

            node_in_count = node[0]
            node_out_count = -1
            node_af = self.__activation_func(node[1])
        nn_param["lh"] = F.Linear(node_in_count, output_dim)
        af_param["lh"] = node_af
        seq.append("lh")
        af_param["lo"] = self.__activation_func(output_af)
        return [nn_param, af_param, seq]

    def __init__(self,input_dim, output_dim, input_af, output_af, hiddens):
        nn_form, af, seq = self.__create_init_param_dict(input_dim,output_dim,input_af,output_af,hiddens)
        super(nn, self).__init__(
            **nn_form
        )
        self.af = af
        self.nn = nn_form
        self.seq = seq

    def __call__(self, x):

        u = x
        for node_name in self.seq:
            if node_name in self.af:
                z = self.af[node_name](u)
            else:
                z = u
            u = self.nn[node_name](z)
        return self.af["lo"](u)


class loss(Chain):

    def __init__(self, predictor):
        super(loss, self).__init__(predictor=predictor)
        self.y = None
        self.loss = None
        self.accuracy = None
        self.compute_accuracy = False
        self.train_statistics = NnTrainStatictis()

    def __call__(self, x, t):

        self.y = self.predictor(x)
        self.loss = F.mean_squared_error(t, self.y)
        self.train_statistics.add_statistics(self.loss.data, 0, x, self.y.data, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = compute_vector_similarly(t.data, self.y.data)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


def compute_vector_similarly(t, y):
    #t_norm = np.linalg.norm(t)
    #diff = np.linalg.norm(t - y)
    #return max(0.01, t_norm - diff) / t_norm * 100.0
    accums = []
    for i in range(len(t)):
        accums.append(compute_scalar_similarly(t[i], y[i]))
    return np.mean(accums)


def compute_scalar_similarly(t, y):
    return max(0.01, t - math.fabs(t - y)) / t * 100.0


def load_data(path, data_limit):
    ret = []
    if not EXT_OF_FLOAT.match(path) is None:
        type = np.float32
    elif not EXT_OF_INT.match(path) is None:
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


class NnModelDefinition:
    def __init__(self):
        self.statistics = ""
        self.data_warehouse = ""
        self.output = ""
        self.hidden_dim = ""
        self.optim = ""
        self.input_af = ""
        self.hidden_af = ""
        self.hidden2_af = ""
        self.hidden2_af = ""
        self.output_af = ""
        self.epoch = ""
        self.batch_size = ""
        self.data_limit = ""


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

    my_nn = nn(len(data[0]), len(label[0]), "id", "id", [[desc[KEY_HIDDEN_DIM], "id"],[desc[KEY_HIDDEN_DIM], desc[KEY_HIDDEN_AF]]])
    model = loss(my_nn)
    optimizer = get_optimizer(desc[KEY_OPTIM])
    optimizer.setup(model)
    train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(data, label),batch_size=desc[KEY_BATCH_SIZE], shuffle=True)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (desc[KEY_EPOCH], 'epoch'), out=".")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/loss','main/accuracy']))
    trainer.run()

    #model.train_statistics.save(statistics_path)
    #chainer.serializers.save_npz(desc[KEY_OUTPUT], my_nn)
    result = None
    if desc[KEY_DATA_WAREHOUSE] + SUF_E_DATA <> None:
        data = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_E_DATA, 0)
        label = load_data(desc[KEY_DATA_WAREHOUSE] + SUF_E_LABEL, 0)
        result = NnResult()
        for idx, (d, l) in enumerate(zip(data, label)):
            predict = my_nn(np.array([d]))
            result.add_result(d, predict, l)
        #result.save(statistics_path)
    return [model,model.train_statistics, result]
ini = "./nn/gen_nn_05.ini"
model, train_statistics, result = training_nn(load_description(ini))
