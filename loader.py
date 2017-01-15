import numpy as np
import chainer
import math
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.datasets import DictDataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from data_treating.number_image import treatment

import re
import os
from datetime import datetime

#warehouse_base_path = "./data_treating/warehouse"
warehouse_base_path = "C:/mySpace/warehouse"

def load_data_and_label(name, filter_str, data_vectorizer, label_vectorizer, limit = -1):
    datas = []
    labels = []
    filter = re.compile(filter_str)
    data_path = warehouse_base_path + "/" + name
    for count, file in enumerate(os.listdir(data_path)):
        if limit > 0 and count > limit:
            break
        if not filter.match(file):
            continue
        file_path = data_path + "/" + file
        answer_label_path = file_path + ".label"
        with open(file_path) as r:
            datas.append(data_vectorizer(r.readline()))
        with open(answer_label_path) as r:
            labels.append(label_vectorizer(r.readline()))
    return [datas, labels]

