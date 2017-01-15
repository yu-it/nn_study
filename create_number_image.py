from datetime import datetime
import os
import sys
from data_treating.number_image.test_data import create_testdata


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    number_format = sys.argv[1]
    number_from = int(sys.argv[2])
    number_to = int(sys.argv[3])
    fontsize_from = int(sys.argv[4])
    fontsize_to = int(sys.argv[5])
    style_set = "./data_treating/data_suppliments/number_image/" + sys.argv[6]
    number_of_image = int(sys.argv[7])
    output = "./data_treating/warehouse/" + time_stamp
    os.mkdir(output)
    if not os.path.exists(style_set):
        raise Exception("パスがありません")

    if number_from > number_to:
        raise Exception("数値の大小関係がへん")

    create_testdata(number_format, number_from, number_to, fontsize_from, fontsize_to, style_set, number_of_image, output)
    pass
