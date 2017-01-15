import os
import random
from data_treating import common_util
from datetime import datetime

try:
    os.mkdir("tmp")
except Exception:
    pass
train_data ,train_label,eval_data ,eval_label = common_util.get_data_path()

for x in range(0,1000):
    y = pow(x,2)
    if common_util.drawing():
        data_path = train_data
        label_path = train_label
    else:
        data_path = eval_data
        label_path = eval_label
    with open(label_path + "\\y.txt","a") as w:
        w.write(str(y) + "\r\n")
    with open(data_path + "\\x.txt", "a") as w:
        w.write(str(x) + "\r\n")
