import os
from datetime import datetime
import random

def drawing(odds = 0.5):
    return random.random() <= odds

def get_data_path(prefix = "", networkfolder = False):
    if prefix == "":
        if networkfolder:
            time_stamp = "t:/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            time_stamp = "./data_treating/warehouse/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    else:
        if networkfolder:
            time_stamp = "t:/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            time_stamp = "./data_treating/warehouse/" + prefix + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    os.mkdir(time_stamp)
    eval_data = time_stamp + "/eval_data"
    os.mkdir(eval_data)
    eval_label = time_stamp + "/eval_label"
    os.mkdir(eval_label)
    train_data = time_stamp + "/train_data"
    os.mkdir(train_data)
    train_label = time_stamp + "/train_label"
    os.mkdir(train_label)
    print ("basepath:" + time_stamp)
    return [train_data, train_label, eval_data, eval_label]