from matplotlib import pyplot
from matplotlib.image import imread
import os
import finger

path = 'C:/FingerprintCNN/Datasets/NISTDB4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'

pngfiles = []
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            pngfiles.append(os.path.join(r, file))
#for f in pngfiles:
#    print(f)
txtfiles = []
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            txtfiles.append(os.path.join(r, file))
#for f in txtfiles:
#    print(f)
mylines = []
with open(txtfiles[0], 'rt') as myfile:
    for myline in myfile:
        mylines.append(myline.rstrip('\n'))
print(mylines)

mylines[0][8:9]
mylines[1][7:9]
mylines[2][9:33]