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


class Consts:
    learning_data_count = 3000
    batch_size = 2
    epoch = 8
    dataset_name =  "C:\\Users\\yuusuke.ito\\Dropbox\\develop\\repositories\\nn\\data_treating\\warehouse\\2016_12_29_15_18_24"
    data_limit = -1
    nn_x_width = 100
    nn_x_height = 100
    nn_conv1_out = 5
    nn_conv2_out = 50
    nn_kernel_size = 5
    nn_l2_per_l1 = 0.5
    output_size = 1000
    label_min = 0.01
    label_max = 0.99


class Encoder(Chain):

    def __init__(self):

        conv1_output_width = (Consts.nn_x_width - (Consts.nn_kernel_size - 1)) / 2

        #l1_input_size = 2490
        #l2_input_size = 1200
        #output_size = 512
        l1_input_size = 1000
        l2_input_size = 512
        l3_input_size = 512
        output_size = 1

        super(Encoder, self).__init__(
            #conv1=F.Convolution2D(1, Consts.nn_conv1_out, (Consts.nn_kernel_size,1)),
            l1=F.Linear(l1_input_size, l2_input_size),
            l2=F.Linear(l2_input_size, output_size),
            #l3 = F.Linear(l3_input_size, output_size)
        )
    def __call__(self, x, batch_size):
        #x = chainer.functions.reshape(x, (batch_size, 1, -1, 1))
        #conv_res = self.conv1(x)
        #conv1_out = F.max_pooling_2d(conv_res, 2)
        #u1 = self.l1(conv1_out)
        u1 = self.l1(x)
        z1 = F.dropout(u1, train=True)
        u2 = self.l2(z1)
        #z2 = F.dropout(u2, train=True)
        #u3 = self.l3(z2)

        return F.sigmoid(u2)
        #return u2

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

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(x, Consts.batch_size)
        self.loss = F.mean_squared_error(self.y, t)
        self.total_loss.append(self.loss.data)
        reporter.report({'loss': self.loss}, self)
        #print('loss:' + str(self.loss.data))
        if self.compute_accuracy:
            self.accuracy = self.y
            self.total_accuracy.append(self.accuracy.data.sum())
            #print('acc:' + str(self.accuracy.data.sum()))
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

my_encoder = Encoder()
from datetime import datetime
time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
import os

os.mkdir(".\\tmp\\" + time_stamp + "\\")

model = loss(my_encoder)
#model = L.Classifier(my_cnn)
optimizer = optimizers.SGD()
optimizer.setup(model)
datas,labels, names,flgs = treatment.load_data_and_label(Consts.dataset_name, Consts.learning_data_count, 0, Consts.output_size,Consts.label_min,Consts.label_max)
edatas,elabels, enames,eflgs = treatment.load_data_and_label(Consts.dataset_name,800,3001, Consts.output_size,Consts.label_min,Consts.label_max)
train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(datas, flgs),batch_size=Consts.batch_size, shuffle=True)
#train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(labels, labels),batch_size=10, shuffle=True)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (Consts.epoch, 'epoch'), out='result')
#trainer.extend(extensions.Evaluator(eval_iter, model))
#trainer.extend(extensions.Evaluator(train_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'loss', 'accuracy']))
#trainer.extend(extensions.ProgressBar())
trainer.run()
plt.clf()
plt.plot(model.total_loss)
plt.savefig(".\\tmp\\" + time_stamp + "\\loss.png")
plt.clf()
plt.plot(model.total_accuracy)
plt.savefig(".\\tmp\\" + time_stamp + "\\acc.png")

for data ,label, name in zip(edatas,elabels, enames):
    encoded = my_encoder(np.array([data]),1)
    #encoded = my_encoder(np.array([label]))
    print(name + ":" + str(encoded.data[0].mean()) + "," + str(encoded.data[0].max()))
    plt.clf()
    plt.ylim([-1.00, 1.00])
    plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    plt.plot(encoded.data[0])
    plt.savefig(".\\tmp\\" + time_stamp + "\\" + name + ".png")
    #
"""
    plt.show()
    plt.plot(label)
    plt.show()
    plt.plot(data)
    plt.show()
"""


