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
import itertools

PATTERN_OF_STR_ARRAY = re.compile(r"[^\*].+")
PATTERN_OF_SEQ_ARRAY = re.compile(r"\*.+")


def get_combo_definition(layer_from, layer_to, node_from, node_to, node_step):
    layers = [[0] + [xx for xx in range(node_from, node_to, node_step)] for x in range(layer_to)]
    return layers


def get_array_from_string(str):
    if PATTERN_OF_SEQ_ARRAY.match(str):
        frm, to, step = str[1:].split()
        to += 1
        return [x for x in range(frm, to, step)]

    else:
        return str.split(",")
    pass

def get_combo(combo, zorome = True):
    ret = []
    for aa in itertools.product(*combo):
        aa = list(aa)
        aa.reverse()

        if not 0 in aa:
            ret.append(aa)
            # print('_________')
            continue
        tmp = aa[aa.index(0) + 1:]
        if len(tmp) > 0 and max(tmp) > 0:
            pass
        elif np.sum(aa) == 0:
            pass
        elif aa.count(aa[0]) + aa.count(0) <> len(aa) and zorome:   #とりあえず今はすべての層のノード数をそろえることとする。
            pass
        elif aa in ret:
            pass
        else:
            ret.append(aa)
            # print('_________')
            pass

    return ret
layer_from = 2
layer_to = 5
node_from = 10
node_to = 15
node_step = 2


combination_definition = get_combo_definition(layer_from, layer_to, node_from, node_to, node_step)
combination = get_combo(combination_definition)
af_combination_definition = get_combo_definition(layer_from, layer_to, 0, 3, 1)
af_combination = get_combo(af_combination_definition,False)
af_and_hidden_definition = []
for x in itertools.product(combination, af_combination):
    add = True
    for i in range(len(x[0])):
        if (x[0][i] == 0 and x[1][i] > 0) or (x[0][i] > 0 and x[1][i] == 0):
            add = False


    if add:
        af_and_hidden_definition.append(x)
print af_and_hidden_definition