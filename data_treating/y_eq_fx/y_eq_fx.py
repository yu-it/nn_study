import os
import random
from datetime import datetime

try:
    os.mkdir("tmp")
except Exception:
    pass
time_stamp = "tmp\\" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

os.mkdir(time_stamp)
eval_data = time_stamp + "\\eval_data"
os.mkdir(eval_data)
eval_label = time_stamp + "\\eval_label"
os.mkdir(eval_label)
train_data = time_stamp + "\\train_data"
os.mkdir(train_data)
train_label = time_stamp + "\\train_label"
os.mkdir(train_label)

for x in range(0,1000):
    y = pow(x,2)
    if random.randint(1,2) == 1:
        data_path = train_data
        label_path = train_label
    else:
        data_path = eval_data
        label_path = eval_label
    with open(label_path + "\\y.txt","a") as w:
        w.write(str(y) + "\r\n")
    with open(data_path + "\\x.txt", "a") as w:
        w.write(str(x) + "\r\n")
