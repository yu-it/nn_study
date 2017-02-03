import numpy as np
import math
import os
import data_treating.common_util as common_util
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

try:
    os.mkdir("tmp")
except Exception:
    pass
train_data ,train_label,eval_data ,eval_label = common_util.get_data_path("mnist")
exs=[]
eys=[]
txs=[]
tys=[]
for idx,(x,  y) in enumerate(zip(mnist["data"], mnist["target"])):
    if idx % 1000 == 0:
        print(str(idx) + " is processed")
    if common_util.drawing(5.0/6.0):
        continue
    if common_util.drawing(0.9):
        xs = txs
        ys = tys
    else:
        xs = exs
        ys = eys
    ys.append(",".join(["1" if i == y else "0" for i  in range(10)]))
    xs.append(",".join([str(xw) for xw in x]))

with open(train_label + "\\y.txt","a") as w:
    w.write("\n".join(tys))
    #w.writelines(tys)
with open(train_data + "\\x.txt", "a") as w:
    w.write("\n".join(txs))
    #w.writelines(txs)
with open(eval_label + "\\y.txt","a") as w:
    w.write("\n".join(eys))
    #w.writelines(eys)
with open(eval_data + "\\x.txt", "a") as w:
    w.write("\n".join(exs))
    #w.writelines(exs)
