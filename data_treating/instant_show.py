import numpy 
from matplotlib import pylab as plt
f = "C:/Users/yuusuke.ito/Dropbox/develop/repositories/nn/data_treating/warehouse/2016_12_03_16_04_30/white.jpg_1.reg.1.csv"
with open(f) as r:
    source = r.readline()

list = source.split(",")
list = [float(x) for x in list]
matrix = numpy.reshape(list,[100,100])
plt.imshow( matrix )
plt.show( matrix )
