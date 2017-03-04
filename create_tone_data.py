# -*- coding: utf-8 -*-

from datetime import datetime
import os
import sys
from data_treating.tone_data.test_data import create_testdata


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    min_tone = int(sys.argv[1])
    max_tone = int(sys.argv[2])
    polytone_count = int(sys.argv[3])
    number_of_tone_color = int(sys.argv[4])
    #style_set = "./data_treating/data_suppliments/tone_data/" + sys.argv[5]
    output = "./data_treating/warehouse/" + time_stamp
    os.mkdir(output)
    #if not os.path.exists(style_set):
    #    raise Exception("パスがありません")

    #create_testdata(min_tone, max_tone, polytone_count, number_of_tone_color, style_set, output)
    create_testdata(min_tone, max_tone, polytone_count, number_of_tone_color, "", output, 00)   #style_set用途未定のため

pass

def show(path, num):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    ary = sorted(os.listdir(path))
    with open(ary[num]) as w:
        npary = np.asarray(w.read().split(","),dtype=np.float32)
    plt.plot(npary)
    plt.show()
