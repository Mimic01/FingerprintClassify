from matplotlib import pyplot
from matplotlib.image import imread
import os
from os import listdir, makedirs
import finger
from finger import Finger
from finger import fingerMetadata
from numpy import load
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from random import random, seed
from shutil import copyfile

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

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     filename = dactList[i].pic
#     image = imread(filename)
#     pyplot.imshow(image)
# pyplot.show()

# script to create test and train subdirectories with copies from dataset for the flow_from_directory API
dataset_home = 'API_dataset_NISTDB4/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    labeldirs = ['A', 'L', 'R', 'T', 'W']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

seed(1)
val_ratio = 0.25
for file in pngfiles:
    src = file
    pngFileName = file[97:109]
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    for dact in dactList:
        if dact.typeclass == 'A':
            dst = dataset_home + dst_dir + 'A/' + pngFileName
            print("Copying {0} to {1}/A/ directory... ", pngFileName, dst_dir)
            copyfile(src, dst)
            break
        elif dact.typeclass == 'L':
            dst = dataset_home + dst_dir + 'L/' + pngFileName
            print("Copying {0} to {1}/L/ directory... ", pngFileName, dst_dir)
            copyfile(src, dst)
            break
        elif dact.typeclass == 'R':
            dst = dataset_home + dst_dir + 'R/' + pngFileName
            print("Copying {0} to {1}/R/ directory... ", pngFileName, dst_dir)
            copyfile(src, dst)
            break
        elif dact.typeclass == 'T':
            dst = dataset_home + dst_dir + 'T/' + pngFileName
            print("Copying {0} to {1}/T/ directory... ", pngFileName, dst_dir)
            copyfile(src, dst)
            break
        elif dact.typeclass == 'W':
            dst = dataset_home + dst_dir + 'W/' + pngFileName
            print("Copying {0} to {1}/W/ directory... ", pngFileName, dst_dir)
            copyfile(src, dst)
            break

