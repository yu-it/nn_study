import PIL
import numpy


def serialize(image):
    array = numpy.array(image)
    maxcol, maxrow = image.size
    ret = []
    for y in range(maxrow):
        for x in range(maxcol):
            ret.append(str(array[y, x][0]))
            ret.append(str(array[y, x][1]))
            ret.append(str(array[y, x][2]))
    return ",".join(ret)

def regularize1(image):
    array = numpy.array(image)
    maxcol, maxrow = image.size
    ret = []
    for y in range(maxrow):
        for x in range(maxcol):
            pixel = 0.0
            pixel += float(array[y, x][0]) / float(255)
            pixel += float(array[y, x][1]) / float(255)
            pixel += float(array[y, x][2]) / float(255)
            ret.append(str(1- round(pixel / float(3), 5)))
    return ",".join(ret)


def regularize2(image):
    array = numpy.array(image)
    maxcol, maxrow = image.size
    ret = []
    for y in range(maxrow):
        for x in range(maxcol):
            pixel = 0.0
            pixel += float(array[y, x][0]) / float(255)
            pixel += float(array[y, x][1]) / float(255)
            pixel += float(array[y, x][2]) / float(255)
            ret.append(str(1 if round(pixel / float(3), 5) < 0.8 else 0))
    return ",".join(ret)
    pass

regularizers = [regularize1, regularize2]
