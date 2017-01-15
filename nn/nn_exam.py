import numpy as np

import time
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

consts_batch_size = 1
consts_epoch = 500
consts_hidden_layer_node_count = 4
consts_hidden_layer_count = 1

consts_input_vector_size = 2
consts_output_vector_size = 1


class nn(Chain):

    def __init__(self,hidden_layer):
        self.hidden_layer_count = hidden_layer
        super(nn, self).__init__(
            li=F.Linear(consts_input_vector_size, consts_hidden_layer_node_count),
            lh=F.Linear(consts_hidden_layer_node_count,consts_hidden_layer_node_count),
            lo=F.Linear(consts_hidden_layer_node_count, consts_output_vector_size)
        )
    def __call__(self, x):
        u = self.li(x)
        #z1 = F.dropout(F.sigmoid(u1), train=True)
        z = u
        u = self.lh(z)
        z = F.relu(u)

        uo = self.lo(z)

        return uo
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
        self.y = self.predictor(x)
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

def get_data_and_label_detail():
    return[
        [0,0,0,1,1,0,1,1],
        [0,1,1,0]
    ]
    #
    """
    data = []
    label = []
    for x in xrange(1000):
        if x % 2 == 1:
            data.append(x)
            label.append(math.pow(x, 2))
    return [data, label]
    """
def get_eval_data_and_label_detail():
    return[
        [0,0,0,1,1,0,1,1],
        [0,1,1,0]
    ]
    #
    """
    data = []
    label = []
    for x in xrange(1000):
        if x % 2 == 1:
            data.append(x)
            label.append(math.pow(x, 2))
    return [data, label]
    """

def get_data_and_label():
    data,label=get_data_and_label_detail()

    data = np.reshape(np.array(data, dtype=np.float32), [len(data) / consts_input_vector_size,consts_input_vector_size])
    label = np.reshape(np.array(label, dtype=np.float32), [len(label) / consts_output_vector_size,consts_output_vector_size])

    return [data,
           label]
    pass


def get_eval_data_and_label():
    data,label=get_eval_data_and_label_detail()

    data = np.reshape(np.array(data, dtype=np.float32), [len(data) / consts_input_vector_size,consts_input_vector_size])
    label = np.reshape(np.array(label, dtype=np.float32), [len(label) / consts_output_vector_size,consts_output_vector_size])

    return [data,
           label]
    pass


nn = nn(consts_hidden_layer_count)
model = loss(nn)

optimizer = optimizers.SGD()
optimizer.setup(model)
datas,labels = get_data_and_label()
train_iter = iterators.SerialIterator(tuple_dataset.TupleDataset(datas, labels),batch_size=consts_batch_size, shuffle=True)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (consts_epoch, 'epoch'), out="c:\\myspace\\my_result")
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch']))

trainer.run()

edatas,elabels =  get_eval_data_and_label()
answers = []
answers2 = []
for x in edatas:
    a = nn(np.reshape(x,[1,1,consts_input_vector_size]))
    answers.append(a.data[0][0])
    #answers2.append(a.data[0][1])

plt.clf()
#plt.plot(edatas,answers)
plt.plot(answers)
#plt.plot(answers2)
#plt.plot(np.reshape(elabels,[len(elabels)]))
plt.show()
print(edatas)
print("------------")
print(answers)
