# -*- coding: utf-8 -*-

from datetime import datetime
from string import  Template
from data_treating import common_util
import random
import os
insurer_base_order = 1000
rece_count_base_order = insurer_base_order * 10
expense_base_order = rece_count_base_order * 1000


train_data, train_label, eval_data, eval_label = common_util.get_data_path("csv_sparse")
head1 = "#1件数A,1費用B,1被保険者数C,1B/A,1B/C," \
        "2件数A,2費用B,2被保険者数C,2B/A,1B/C," \
        "3件数A,3費用B,3被保険者数C,3B/A,3B/C,"
head_label = "T件数A,T費用B,T被保険者数C,TB/A,TB/C,"
for data_index in range(5000):
    insurer_acc = 0
    rece_acc = 0
    exp_acc = 0
    record = ""
    for acc_count in range(3):

        insurer = int(insurer_base_order * random.uniform(0.5,1.5))
        rece = int(rece_count_base_order * random.uniform(0.5,1.5))
        expense = int(expense_base_order * random.uniform(0.5,1.5))
        B_div_A = expense / rece
        B_div_C = expense / insurer

        insurer_acc += insurer
        rece_acc += rece
        exp_acc += expense
        record += "," + Template("$a,$b,$c,$d,$e").substitute(a=rece, b=expense, c=insurer, d= B_div_A, e=B_div_C)
    data_path = train_data
    label_path = train_label
    if common_util.drawing():
        data_path = eval_data
        label_path = eval_label

    with open(data_path + "/" + str(data_index) + ".csv","w") as w:
        w.write(head1 + "\r\n")
        w.write(record[1:])

    B_div_A_acc = exp_acc / rece_acc
    B_div_C_acc = exp_acc / insurer_acc

    record = Template("$a,$b,$c,$d,$e").substitute(a=rece_acc, b=exp_acc, c=insurer, d=B_div_A_acc, e=B_div_C_acc)
    with open(label_path + "/" + str(data_index) + ".csv","w") as w:
        w.write(head_label + "\r\n")
        w.write(record)

