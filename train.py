import cv2
import numpy as np
import utils
import os
import sys

# SUPPORT VECTOR MACHINE PARAMETERS
# -> POLY kernel
poly_params = dict ( 
    kernel = cv2.ml.SVM_LINEAR,
    svm_type    = cv2.ml.SVM_C_SVC,
    C           = 2.67,
    gamma       = 5.383,
    degree      = 5
)

# calculate HOGs for a list of images
# and acumulate those in a python list
def build_input(filelist, hog_desc):
    data           = []
    labels         = []
    width,height   = hog_desc.winSize
    print(width, height)
    
    print "[1/4] reading data..."
    ii = 0
    for name, label in filelist:
        sys.stdout.write("\r" + str(ii) + "/" + str(len(filelist)))
        img     = cv2.imread(name)
        img     = cv2.resize(img, (width, height))
        imgHOG  = utils.calculate_HOG(img, hog_desc)
        
        data.append(imgHOG)
        labels.append(label)
        ii += 1
        
    print
    print "[2/4] processing data..."
    size    = len(data[0])
    data    = np.float32(data).reshape(-1, size)
    labels  = np.int32(labels).reshape(-1, 1)
    return data, labels
    
def train(positive, negative, hog_desc, svm_params=None, svm_filename='svm_data.dat'):

    if svm_params is None:
        svm_params = poly_params
    positive = utils.get_files(positive)
    negative = utils.get_files(negative)

    positive = utils.label_list(positive, 1)
    negative = utils.label_list(negative, 0)

    files = positive + negative

    trainData, responses = build_input(files, hog_desc)

    print "[3/4] training SVM..."
    svm = cv2.ml.SVM_create()
    vm = cv2.ml.SVM_create()
    svm.setC(2.67)
    svm.setDegree(5)
    svm.setGamma(5.383)
    svm.setKernel(cv2.ml.SVM_POLY)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)

    # print(trainData[1,:].shape)
    # print(trainData[1:].dtype)
    
    print "[4/4] saving SVM to a file..."
    svm.save("Auta/data.dat")

   
    with open(os.path.splitext(svm_filename)[0] + '.desc', 'w') as f:
        s = 'hog_desc: winSize=%s, blockSize=%s, blockStride=%s, cellSize=%s, nbins=%s'
        f.write(s % (str(hog_desc.winSize), str(hog_desc.blockSize), str(hog_desc.blockStride), str(hog_desc.cellSize), str(hog_desc.nbins)))

    return svm
