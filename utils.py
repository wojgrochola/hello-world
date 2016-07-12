import cv2
import numpy as np
import glob
import os
import detect

def calculate_HOG(img, hog):
    """
        calculates Histogram of Gradients
        and returns values as a python list

        img - 
        params = (winSize, blockSize, blockStride, cellSize, nbins)
    """
    # hog = cv2.HOGDescriptor(*params)
    res = hog.compute(img)
    res = res.reshape(-1, len(res))
    return res[0].tolist()

def get_files(path):
    """
        returns list of file names from path
    """
    if path[-1] != '/':
        path += '/'
    path = path + '*.*'
    filenames = [f for f in glob.iglob(path)]
    return filenames

def label_list(List, label):
    """
        labels each item from the list
        returns list of pairs of items form list and label
    """
    labels  = [label for i in xrange(len(List))]
    labeled = zip(List, labels)
    return labeled

def cut_out(path):
    print path
    image = cv2.imread(path)
    img = image[0:32,8:72]
    cv2.imwrite('top' + os.path.basename(path), img)
    img = image[32:64,8:72]
    cv2.imwrite('bottom' + os.path.basename(path), img)
    img = image[8:56,8:32]
    cv2.imwrite('left' + os.path.basename(path), img)
    img = image[8:56,48:72]
    cv2.imwrite('right' + os.path.basename(path), img)

def prepare(path, p1, p2, ii):
    img = cv2.imread(path)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = min(dx / 3, p1[0], img.shape[1] - 1 - p2[0])
    x0 = p1[0] - d
    x1 = p2[0] + d

    d = 2 * dx / 5 + 4 * d / 5 - dy / 2
    y0 = p1[1] - d
    y1 = p2[1] + d

    if y0 < 0:

        d = min(dy / 3, p1[1], img.shape[0] - 1 - p2[1])
        y0 = p1[1] - d
        y1 = p2[1] + d
        d = 5 * dy / 8 + 5 * d / 4 - dx / 2
        x0 = p1[0] - d
        x1 = p2[0] + d

    img = img[y0:y1 + 1, x0:x1 + 1]
    if img is not None:
        img = cv2.resize(img, (80,64))
    cv2.imwrite(ii + '_' + os.path.basename(path), img)

def rect(path, p1, p2):
    img = cv2.imread(path)
    cv2.rectangle(img, p1, p2, (0,0,255), thickness=2)
    cv2.imwrite(path, img)

def reflect(path):
    for f in get_files(path):
        img = cv2.imread(f)
        img = cv2.flip(img, 1)
        cv2.imwrite(os.path.splitext(f)[0] + '_flipped.jpg', img)

def convert_with_origin(origin, obj):
    obj = detect.DetectedObject(obj.image,tuple(map(sum,zip(obj.origin,origin))),obj.size)
    return obj

def save():
    files = get_files('../video/bikes/detected/video')
    img = cv2.imread(files[0])
    video = cv2.VideoWriter('video.avi',-1,1,img.shape[:2])
    i = 0
    for f in files:
        print i
        i += 1
        img = cv2.imread(f)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    """
    #reflect(r'D:\SVM\Cars\Rear-end\v2\BL\In\Positive')
    #prepare(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (385,458), (579,615), '0')
    #prepare(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (658,436), (827,592), '1')
    #prepare(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (924,451), (1121,609), '2')
    #prepare(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (77,478), (264,606), '3')
    rect(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (74,460), (264,610))
    rect(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (370,450), (570,625))
    rect(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (650,435), (835,600))
    rect(r'D:\SVM\prezentacja\Out\IMG_2771.JPG', (905,448), (1130,630))
    """
    #rect(r'D:\SVM\prezentacja\Out\Done\full_uncut\1 - Kopia.jpg', (827,380), (990,503))
    #rect(r'D:\SVM\prezentacja\Out\Done\full_uncut\1 - Kopia.jpg', (422,390), (565,504))

    #for f in get_files('.'):
    #    cut_out(f)
    save()
        
