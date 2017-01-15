# -*- coding: utf-8 -*-
import scipy
import os
import re
import math
import numpy
from scipy.io.wavfile import read

#最大値に始まり、正規化された数値による配列に変換する。
def normalization(wav_file,array_length, offset):
    rtn = numpy.zeros(array_length, dtype=numpy.float32)
    fs,wav = read(wav_file)
    wav = numpy.array(wav, dtype=numpy.float32)
    tmp_partial_array = wav[offset:offset + 5 * (array_length)]
    max_val = max(tmp_partial_array)
    idx = 0
    while tmp_partial_array[idx] <> max_val:
        idx += 1

    partial_array = wav[idx:idx + array_length]

    rtn[0:array_length] = partial_array #partial_arrayがarray_lengthより短い場合に。

    #正規化
    #標準偏差
    normalized_rtn = normalize_array(rtn)
    return [rtn, normalized_rtn]


def gaussian_normalize_array(rtn):

    rtn -= rtn.mean()
    sd = numpy.sqrt(((rtn - rtn.mean()) ** 2).sum() / len(rtn))
    rtn /= sd
    #rtn /= rtn.max()
    return rtn


def normalize_array(rtn):

    return (rtn - rtn.min()) / (rtn.max() - rtn.min())

path_dict = None


def lookup_file(path, num):
    global path_dict
    if path_dict == None:
        path_dict = dict()
        for wav in os.listdir(path):
            if re.match(".+.wav", wav):
                path_dict[wav.split("_")[0]] = wav.replace(".wav","")
    return path_dict[str(num)]

def load(path, num, label_size, min, max):
    wav_file = lookup_file(path,num)
    fname_prefix = path + "/" + wav_file.replace(".wav","")
    ret = dict()
    with open(fname_prefix + "_nor.csv") as r:
        ret["normalized"] = numpy.array(r.readline().split(","), dtype=numpy.float32)

    with open(fname_prefix + "_raw.csv") as r:
        ret["raw"] = numpy.array(r.readline().split(","), dtype=numpy.float32)

    with open(fname_prefix + "_vector.txt") as r:
        ret["vector"] = numpy.array(r.readline().split(","), dtype=numpy.int32)

    with open(fname_prefix + "_tone.txt") as r:
        ret["tones"] = numpy.array(r.readline().split(","), dtype=numpy.int32)

    answers = []
    answers2 = []
    for x in ret["tones"]:
        #answers.append(get_sin_wave(x / 12 + 1, len(ret["normalized"])))
        answers.append(get_sin_wave(x / 12, label_size, min, max))
        answers2.append(get_sin_wave_by_tone(44100, available_tones[x],label_size))
    ret["fft"] = numpy.array(scipy.fft(ret["normalized"]),dtype=numpy.float32)
    ret["sin_waves"] = answers
    ret["sin_waves2"] = answers2
    ret["name"] = wav_file
    return ret
def get_sin_wave_by_tone(samplingrate, name, length):
    freq = get_frequency(name.replace("#","X"))
    time = length * (1.0 / samplingrate)
    number_of_wave = freq / (1 / time)
    ret = []
    for i in xrange(length):
        val = (math.sin(degree_to_radian(((number_of_wave * 360.0) / length) * float(i)) % 360) + 1) / 2
        ret.append(val * 0.98 + 0.01)
    return numpy.array(ret, dtype=numpy.float32)


available_tones = ["C1","C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2","D#2","E2","F2","F#2","G2","G#2","A2","A#2","B2","C3","C#3","D3","D#3","E3","F3","F#3","G3","G#3","A3","A#3","B3","C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4","C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5","B5","C6","C#6","D6","D#6","E6","F6","F#6","G6","G#6","A6","A#6","B6","C7","C#7","D7","D#7","E7","F7","F#7","G7","G#7","A7","A#7","B7"]

freq_of_C1 = 32.70319566
freq_dict = {"C":32.7031956629012,
            "CX":34.6478288724778,
            "D":36.7080959900911,
            "DX":38.8908729657258,
            "E":41.2034446146295,
            "F":43.6535289297063,
            "FX":46.2493028396004,
            "G":48.9994294984358,
            "GX":51.9130871982874,
            "A":55.0000000008781,
            "AX":58.2704701907303,
            "B":61.7354126580833
            }
def get_frequency(name):
    tone_name = name[0:1] if len(name) == 2 else name[0:2]
    tone_height = float(name[-1])
    return freq_dict[tone_name] * (math.pow(2,tone_height - 1))





def load_data_and_label(path, count, offset,label_size, min, max, tone, option):
    datas = []
    labels = []
    names = []
    exact_match_flgs = []
    flgs = []
    octave = []
    octave2 = []
    ffts = []
    labels2 = []
    for x in range(count):
        data = load(path,x + offset,label_size, min, max)
        datas.append(data["normalized"])
        labels.append(data["sin_waves"][0] if data["tones"][0] == tone else numpy.zeros(label_size, dtype=numpy.float32))
        names.append(data["name"])
        ffts.append(data["fft"])
        flg_tone_is_match = 0.99 if data["tones"][0]  % 12 == tone % 12 else 0.01
        exact_match_flgs.append(numpy.array([0.99 if data["tones"][0] == tone else 0.01], dtype=numpy.float32))
        flgs.append(numpy.array([flg_tone_is_match], dtype=numpy.float32))
        octave.append(numpy.array([flg_tone_is_match, math.fabs(option - (data["tones"][0] / 12)) + 1], dtype=numpy.float32))
        octave2.append(numpy.array([math.fabs(option - (data["tones"][0] / 12)) + 1 if flg_tone_is_match > 0.5 else 0.01], dtype=numpy.float32))
        labels2.append(data["sin_waves2"][0] if data["tones"][0] % 12 == tone % 12 else numpy.zeros(label_size, dtype=numpy.float32))

    return [datas, labels, names,exact_match_flgs, flgs, octave ,octave2, labels2, ffts]
def get_sin_wave(count, width, min, max):
    count = math.pow(2,count)
    rate=2
    ary = [(max-min) * (math.sin(1.5708 + (2.0 * math.pi) * (count * float(x/rate) / float(width)))+ 1.0) / 2 + min for x in range(width)]

    return numpy.array(ary, dtype=numpy.float32)


def degree_to_radian(degree):
    return degree * math.pi / 180


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ary = get_sin_wave_by_tone(44000, "A4", 100)
    plt.plot(ary)
    plt.show()
    #
    """
    import matplotlib.pyplot as plt
    ary = get_sin_wave(8, 512, 0.01, 0.99)
    print(ary.min())
    print(ary.max())
    plt.plot(ary)
    plt.show()

    pass
    data= load("C:\\Users\\yuusuke.ito\\Dropbox\\develop\\repositories\\nn\\data_treating\\warehouse\\2016_12_29_15_18_24",1)
    import matplotlib.pyplot as plt
    plt.plot(data["normalized"])
    plt.show()

    plt.plot(data["raw"])
    plt.show()

    data = load("C:\\Users\\yuusuke.ito\\Dropbox\\develop\\repositories\\nn\\data_treating\\warehouse\\2016_12_29_15_18_24",
                2)
    import matplotlib.pyplot as plt

    plt.plot(data["normalized"])
    plt.show()

    plt.plot(data["raw"])
    plt.show()

#    fs, wav = read("C:\\Users\\yuusuke.ito\\Dropbox\\develop\\repositories\\nn\\data_treating\\warehouse\\2016_12_29_15_18_24\\1.wav")
#    plt.plot(wav[1000:2000])
#    plt.show()
"""
