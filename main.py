import train, detect
import cv2
import numpy as np
import utils
import os
from matplotlib import pyplot as plt

SVM_PATH = '../Data/'

def test(path, svm_name, hog_desc):
	tests = path
	out = path + '/Out/'
	svm = SVM_PATH + svm_name

	detect.test(tests, hog_desc, out, svm_path=svm)

if __name__ == '__main__':
	hog_desc = cv2.HOGDescriptor((128,128), (16,16), (8,8), (8,8), 9)
	path = 'Auta/'
	positive = path + 'Positives/'
	negative = path + 'Negatives3/'
	tests = path + 'Test4/'
	out = path + 'Out/'
	svm = train.train(positive, negative, hog_desc)
	detect.test(svm, tests, hog_desc, out, 2)
   



	
