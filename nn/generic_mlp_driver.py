# -*- coding: utf-8 -*-
from datetime import datetime
import nn.generic_mlp as generic_mlp
import numpy as np
from chainer import serializers
import re
import itertools
import os

TEMPLATE = """
[path]
statictis = {statictis}
data_path = {data_path}

[model]
hidden = {hidden_form}
activation_input = id
activation_output = {output_activation}
optimizer={optimizer}
value_type={val_type}

[training]
epoch={epoch}
batch_size={batch_size}
data_limit={data_limit}
"""





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

        if 0 in aa:
            tmp = aa[aa.index(0) + 1:]
            if len(tmp) > 0 and max(tmp) > 0:
                continue
            elif np.sum(aa) == 0:
                continue
        #elif aa.count(aa[0]) + aa.count(0) <> len(aa) and zorome:  # とりあえず今はすべての層のノード数をそろえることとする。
        if (aa.count(aa[0]) + aa.count(0)) <> len(aa) and zorome:  # とりあえず今はすべての層のノード数をそろえることとする。
            continue
        if aa in ret:
            continue
        ret.append(aa)

    return ret


def check_and_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path




if __name__ == "__main__":
    import sys
    import ConfigParser

    conf = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]  #0 client,1 server
    else:
        mode = 0
    generic_mlp.MODE = mode
    print(conf)
    inifile = ConfigParser.SafeConfigParser()
    inifile.read(conf)

    #各種変数定義

    optimizer_from = 0
    optimizer_to = 1

    layer_from = inifile.getint("combination","layer_from")
    layer_to = inifile.getint("combination","layer_to")
    node_from = inifile.getint("combination","node_from")
    node_to = inifile.getint("combination","node_to")
    node_step = inifile.getint("combination","node_step")
    af_same_all = inifile.get("combination","af_same_all")

    epoch_from = inifile.getint("combination","epoch_from")
    epoch_to = inifile.getint("combination","epoch_to")
    epoch_step = inifile.getint("combination","epoch_step")

    batch_size_from = inifile.getint("combination","batch_size_from")
    batch_size_to = inifile.getint("combination","batch_size_to")
    batch_size_step = inifile.getint("combination", "batch_size_step")

    activation_funcs = [""]
    optimizer_funcs = []
    activation_funcs.extend(inifile.get("combination", "activations").split(","))
    optimizer_funcs.extend(inifile.get("combination", "optimizers").split(","))

    hidden_activation_from = 0
    hidden_activation_to = len(activation_funcs)

    epochs = [epoch_from + epoch_step * (i) for i in range(((epoch_to - epoch_from) / epoch_step) + 1)]
    batch_sizes = [batch_size_from + batch_size_step * (i) for i in range(((batch_size_to - batch_size_from) / batch_size_step) + 1)]
    optimizers = range(optimizer_from, len(optimizer_funcs))


    hidden_dim_combination_definition = get_combo_definition(layer_from, layer_to, node_from, node_to, node_step)
    hidden_dim_combination = get_combo(hidden_dim_combination_definition)
    hidden_af_combination_definition = get_combo_definition(layer_from, layer_to, hidden_activation_from, hidden_activation_to, 1)
    hidden_af_combination = get_combo(hidden_af_combination_definition, af_same_all.lower() == "true")
    hidden_af_and_hidden_dim_definition = []
    for x in itertools.product(hidden_dim_combination, hidden_af_combination):
        add = True
        for i in range(len(x[0])):
            if (x[0][i] == 0 and x[1][i] > 0) or (x[0][i] > 0 and x[1][i] == 0):
                add = False


        if add:
            hidden_af_and_hidden_dim_definition.append(x)

    #epoch


    #batch_size


    base_path = check_and_create_path(inifile.get("running_definition", "base"))
    base_path = base_path + "/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(base_path)
    save_fig = check_and_create_path(inifile.get("running_definition", "save_fig"))

    generic_mlp.SAVE_FIG = save_fig.lower() == "true"

    statistics_base_path = check_and_create_path(base_path + "/statistics")
    config_base_path = check_and_create_path(base_path + "/configs")
    model_base_path = check_and_create_path(base_path + "/models")
    evaluation_path = check_and_create_path(base_path) #とりあえずルートにしておく
    statistics = []
    generated_configs = []
    for idx, definition in enumerate(itertools.product(optimizers, batch_sizes, epochs, hidden_af_and_hidden_dim_definition)):

        hidden_form = ""
        for node_count,activation in zip(*definition[3]):
            if node_count > 0:
                hidden_form += "-" + str(node_count) + "," + activation_funcs[activation]

        hidden_form = hidden_form[1:]
        statistics_path =  check_and_create_path(statistics_base_path + "/" + str(idx))
        config = TEMPLATE.format(statictis=statistics_path,
                                 data_path=inifile.get("running_definition", "data_path"),
                                 output_activation=inifile.get("running_definition", "output_activation"),
                                 val_type=inifile.get("running_definition", "value_type"),
                                 data_limit=inifile.get("running_definition", "data_limit"),
                                 optimizer=optimizer_funcs [definition[0]],
                                 batch_size=definition[1],
                                 epoch=definition[2],
                                 hidden_form=hidden_form)
        config_file = config_base_path + "/" + str(idx) + ".ini"
        with open(config_file ,"w") as w:
            w.write(config)
        generated_configs.append([config_file,statistics_path])

    for idx,target in enumerate(generated_configs):
        file = target[0]
        statistics_path = target[1]
        model, train_statistics, result = generic_mlp.training_nn_by_conf(file)

        model_file = model_base_path + "/" + str(idx) + ".model"
        serializers.save_npz(model_file, model)

        result.save(statistics_path)
        train_statistics.save(statistics_path)
        mean_accuracy, loss_accumrate = result.compute_summary_data()
        with open(file) as r:
            config = "\n".join(r.readlines())
        statistics.append((loss_accumrate, idx, config, mean_accuracy,train_statistics))
        print("no{num} by_[{hidden_form}]_epoch:{epoch}_batch:{batch} elapsed:{elapsed}".format(
                                    num=idx,
                                    hidden_form=train_statistics.nn_structure,
                                    epoch=train_statistics.epoch_count,
                                    batch=train_statistics.batch_size,
                                    elapsed=(train_statistics.end_time - train_statistics.start_time).total_seconds()))

    statistics = sorted(statistics, key=lambda v : v[0])

    ranking_file = base_path + "/ranking.txt"
    ranking_detail_file = base_path + "/ranking_detail.txt"
    for idx, s in enumerate(statistics):
        stat = s[4]
        digest_temp = "{rank}_config-no_{conf_no}_{loss}_{accurate}_by_[{hidden_form}]_epoch:{epoch}_batch:{batch} elapsed:{elapsed}"
        #digest = str(idx) + "_config-no_" + str(s[1]) + "_" + str(s[0]) + "_" + str(s[3])
        digest = digest_temp.format(rank=idx,
                                    conf_no=s[1],
                                    loss=s[0],
                                    accurate=s[3],
                                    hidden_form=stat.nn_structure,
                                    epoch=stat.epoch_count,
                                    batch=stat.batch_size,
                                    elapsed=(stat.end_time - stat.start_time).total_seconds())
        with open(ranking_file, "a") as w:
            w.write(digest + "\n")
        with open(ranking_detail_file, "a") as w:
            w.write(digest + "\n")
            w.write(str(s[2]))
            w.write("\n")
            w.write("\n")
            w.write("---------------------------------------------------------------------")
            w.write("\n")

