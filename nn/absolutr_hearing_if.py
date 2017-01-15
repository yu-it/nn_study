# -*- coding: utf-8 -*-
import numpy
import matplotlib.pyplot as plt
import sys
import pyaudio
CHANNELS = 1        #モノラル
RATE = 44100        #サンプルレート
FRAME = 1000       #データ点数

p = pyaudio.PyAudio()
count = p.get_device_count()
devices = []
for i in range(count):
    devices.append(p.get_device_info_by_index(i))
for i, dev in enumerate(devices):
    print (i, dev['name'])
print("選んで")
input = int(sys.stdin.readline())
while input < 0 or input >= count:
    input = int(sys.stdin.readline())
print("listen " + str(input))

audio = pyaudio.PyAudio()
frames = []


def callback(in_data, frame_count, time_info, status):
    global frames
    tmp = frames[:]
    tmp.extend(numpy.fromstring(in_data, numpy.int16))  # この中で別スレッドの処理
    if len(tmp) > 1000:
        frames = tmp[-1000:]
    else:
        frames = tmp

    return (None, pyaudio.paContinue)


stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS,
                    rate=RATE, input=True,
                    input_device_index=input,  # デバイスのインデックス番号
                    frames_per_buffer=FRAME, stream_callback=callback)
print ("recording...")

plt.clf()
plt.ylim([-8000, 8000])
plt.yticks([(x - 4) * 1000 for x in range(9)])
#flg,ax = plt.subplots(1,1)

#field, = ax.plot(0)
while True:
    copied = frames[:]
    data = copied
    #print(dir(field))
    plt.clf()
    plt.ylim([-8000, 8000])
    plt.yticks([(x - 4) * 1000 for x in range(9)])
    plt.plot(data)
    plt.pause(0.1)
print ("finished recording")
