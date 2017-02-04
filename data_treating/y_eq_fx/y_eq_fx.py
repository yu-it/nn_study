import os
import random
from data_treating import common_util
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    os.mkdir("tmp")
except Exception:
    pass
train_data ,train_label,eval_data ,eval_label = common_util.get_data_path("y_eq_ax")
xs=[]
ys=[]
for x in range(1,1000):
    x = x * 0.01 + 0.2
    xs.append(x)
    y = float(1)/x
    ys.append(y)
    if common_util.drawing():
        data_path = train_data
        label_path = train_label
    else:
        data_path = eval_data
        label_path = eval_label
    with open(label_path + "/y.txt","a") as w:
        w.write(str(y) + "\r\n")
    with open(data_path + "/x.txt", "a") as w:
        w.write(str(x) + "\r\n")
plt.plot(ys)
plt.savefig("y_eq_ax.png")
plt.show()