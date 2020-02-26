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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from random import random, seed
from shutil import copyfile
import sys


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
# dataset_home = 'C:/FingerprintCNN/API_dataset_NISTDB4/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
#     labeldirs = ['A', 'L', 'R', 'T', 'W']
#     for labldir in labeldirs:
#         newdir = dataset_home + subdir + labldir
#         makedirs(newdir, exist_ok=True)

# seed(1)
# val_ratio = 0.25
# adv = 0
# for file in pngfiles:
#     src = file
#     pngFileName = file[97:109]
#     dst_dir = 'train/'
#     if random() < val_ratio:
#         dst_dir = 'test/'
#     for dact in dactList[adv:]:
#         if dact.typeclass == 'A':
#             dst = dataset_home + dst_dir + 'A/' + pngFileName
#             print("Copying {} to {} /A/ directory... ".format(pngFileName, dst))
#             copyfile(src, dst)
#             adv += 1
#             break
#         elif dact.typeclass == 'L':
#             dst = dataset_home + dst_dir + 'L/' + pngFileName
#             print("Copying {} to {}/L/ directory... ".format(pngFileName, dst))
#             copyfile(src, dst)
#             adv += 1
#             break
#         elif dact.typeclass == 'R':
#             dst = dataset_home + dst_dir + 'R/' + pngFileName
#             print("Copying {} to {}/R/ directory... ".format(pngFileName, dst))
#             copyfile(src, dst)
#             adv += 1
#             break
#         elif dact.typeclass == 'T':
#             dst = dataset_home + dst_dir + 'T/' + pngFileName
#             print("Copying {} to {}/T/ directory... ".format(pngFileName, dst))
#             copyfile(src, dst)
#             adv += 1
#             break
#         elif dact.typeclass == 'W':
#             dst = dataset_home + dst_dir + 'W/' + pngFileName
#             print("Copying {} to {}/W/ directory... ".format(pngFileName, dst))
#             copyfile(src, dst)
#             adv += 1
#             break
# print("Finished copying images to dataset directory!")

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracyu'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

def run_test_harness():
    model = define_model()
    datagen = ImageDataGenerator(rescale=1.0/255)
    train_it = datagen.flow_from_directory('C:/FingerprintCNN/API_dataset_NISTDB4/train', class_mode='categorical', batch_size=16, target_size=(512, 512))
    test_it = datagen.flow_from_directory('C:/FingerprintCNN/API_dataset_NISTDB4/test', class_mode='categorical', batch_size=16, target_size=(512, 512))
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)
        
run_test_harness()
            
