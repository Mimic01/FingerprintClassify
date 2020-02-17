from matplotlib import pyplot
from matplotlib.image import imread
import os
import finger
from finger import Finger
from finger import fingerMetadata

path = 'C:/FingerprintCNN/Datasets/NISTDB4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'

pngfiles = []
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            pngfiles.append(os.path.join(r, file))
txtfiles = []
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            txtfiles.append(os.path.join(r, file))
mylines = []
metaDataList = []
txtfiles_length = len(txtfiles)
for i in range(txtfiles_length):
    with open(txtfiles[i], 'rt') as myfile:
        for myline in myfile:
            mylines.append(myline.rstrip('\n'))
        metaDataLine = fingerMetadata(mylines[0][8:9], mylines[1][7:9], mylines[2][9:33])
        metaDataList.append(metaDataLine)
        mylines.clear()

dactList = []
for idx, onePic in enumerate(pngfiles):
    nFinger = Finger(onePic, metaDataList[idx].gender, metaDataList[idx].typeclass, metaDataList[idx].history)
    dactList.append(nFinger)
