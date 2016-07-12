import cv2
import numpy as np
import os
import utils
import time
import itertools
import json
from sklearn.cluster import DBSCAN
import imutils
DD = dict(top = 0, bottom = 0, left = 0, right = 0)

class DetectedObject(object):

    """
        class representing detected object
    """

    def __init__(self, image, origin, size):
        
        self.image = image
        self.origin = origin
        self.size = size
        self.scale = 1.0

    def scaled(self, value):

        origin = tuple([value * x for x in self.origin])
        size = tuple([value * x for x in self.size])
        
        obj = DetectedObject(self.image, origin, size)
        obj.scale = self.scale * value
        return obj

    def get_last(self):
        x = self.origin[0] + self.size[0] - 1
        y = self.origin[1] + self.size[1] - 1
        return (x,y)

    def to_int(self):

        origin = tuple([int(round(x)) for x in self.origin])
        size = tuple([int(round(x)) for x in self.size])
        
        obj = DetectedObject(self.image, origin, size)
        obj.scale = self.scale
        return obj

    def __str__(self):

        return 'origin = %s, window = %s' % (str(self.origin), str(self.size))

    def get_center(self):
        return tuple(map(lambda (p,w): p + w/2, zip(self.origin, self.size)))

    def to_JSON(self):
        return dict(
                origin = self.origin,
                size = self.size
            )

class ObjectLinker(object):

    def __init__(self, size, min_dist):
        
        self.size = size
        self.dist = min_dist

    def link(self, objects):
        
        positions = np.zeros(self.size, dtype=np.uint8)
        new       = []
        rr        = xrange(-self.dist, self.dist + 1)
        
        for obj in objects:
            
            x,y = obj.get_center()
            if positions[y,x] == 1:
                continue

            for yy in rr:
                for xx in rr:
                    if positions.shape[0] > y + yy >= 0 and positions.shape[1] > x + xx >= 0:
                        positions[y + yy, x + xx] = 1
                        
            new.append(obj)
        return new

class SKLearnLinker(object):
    
    def __init__(self, min_dist, get_min=False):
        
        self.dist = min_dist
        self.get_min = get_min

    def link(self, objects):

        if len(objects) == 0:
            return []
        
        posList = []
        
        for obj in objects:
            
            c = obj.get_center()
            posList.append(c)

        db = DBSCAN(eps=self.dist, min_samples=1).fit(np.array(posList))
        labels = set (db.labels_)
        NC = n_clusters_ = len(labels) - (1 if -1 in labels else 0)
        
        detected = [ [] for i in range(NC) ]
        for i in range(len(posList)):
            detected [ db.labels_[i] ] . append ( objects[i] )

        new = []

        for d in detected:

            if self.get_min:
                dd = [_d for _d in d]
                
                dd = sorted(dd, key=lambda x: x.scaled)
                dd = dd[:5]
                ddd = dd
                dd = [_d.get_center() for _d in dd]
                zipped = zip(*dd)
                mean = tuple([np.mean(i) for i in zipped])
                mn = np.argmin(map(np.linalg.norm, np.array(dd) - np.array([mean for i in dd])))
                new.append(ddd[mn])
                
                #mx = np.argmin(dd)
                #new.append(d[mx])
            else:
                dd = [_d.get_center() for _d in d]
                zipped = zip(*dd)
                mean = tuple([np.mean(i) for i in zipped])
                mn = np.argmin(map(np.linalg.norm, np.array(dd) - np.array([mean for i in dd])))
                new.append(d[mn])
        
        return new
"""
        for d in detected:
            for dd in d:
                new.append(dd)

        return new
"""

def sliding_window(image, stepSize, windowSize):
    
    """
        generator yielding subimages
        
        stepSize - distance between next windows
        windowSize - size of the moving window
    """
    
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(64, 64)):

    """
        generator yielding pyramid of the scaled images
    """
    
    yield image, 1.0

    factor = 1.0
    while True:
        factor *= scale
        w = int(round(image.shape[1] / factor))
        h = int(round(image.shape[0] / factor))
        dim = (w, h)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
            break

        yield img, factor


def non_max_suppression_slow(boxes, overlapTresh):
    if not len(boxes):
        return []

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    #sort coordinates by right-bottom
    idxs = np.argsort(y2) 

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in xrange(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w*h) / area[j]

            if overlap > overlapTresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick]


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def detect(image, hog_desc, svm, stepSize, verbose=3):

    """
        main function detecting all occurances
        of the object defined by svm

        image - image to process
        hog_params - parameters of hog descriptor
        svm - cv2.SVM
        stepSize - distance
    """

    winSize = hog_desc.winSize
    detected = []
    counter = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=winSize):
        counter+=1
        clone = image.copy()
        cv2.rectangle(clone, (x,y), (x+128,y+128), (0,255,0),2)
        if window.shape[:2] != tuple(reversed(winSize)):
            continue

        imgHOG = np.float32(utils.calculate_HOG(window, hog_desc))
        
        result = svm.predict(imgHOG[None, :])[1][0][0]
        # print(result)

        # cv2.imshow("Auto", clone)
        # cv2.waitKey(1)

        if result != 1.0:
            continue
        
        obj = DetectedObject(window, (x,y), winSize)

        if verbose > 3:
            print 4 * ' ' + 'Object detected: %s' % str(obj)

        detected.append(obj)

    return detected

def detect_multiscale(image, hog_desc, svm, scale=1.5, verbose=0, stepSize=12, linker_get_min=False,
                      scale_min=1, scale_max=20.0):

    winSize   = hog_desc.winSize
    imgSize   = image.shape[:2]
    detected  = []
    linker    = SKLearnLinker(max(winSize), get_min=linker_get_min)
    counter = 0
    for resized,factor in pyramid(image, scale=scale, minSize=winSize):
        print counter
        counter += 1
        if factor < scale_min or factor > scale_max:
            continue
        if verbose > 2:
            print 2 * ' ' + 'Scaled by {:.2f}'.format(factor), resized.shape[:2]
        
        d = detect(resized, hog_desc, svm, stepSize=stepSize, verbose=verbose)
        
        d = [obj.scaled(factor).to_int() for obj in d]
        
        detected.extend(d)

    # detected = linker.link(detected)

    if verbose > 1:
        print ('Found %d object' % len(detected)) + ('s' if len(detected) != 1 else '')

    return detected

##################
#
#   TEST
#
##################

def test(svm, directory, hog_desc, out_path, verbose=0):
    name_counter = 0
    
    print(svm)
    if verbose > 0:
        print 'Loading SVM...'
    # svm = cv2.ml.SVM_create()
    # svm.load(svm_path)
    # cv2.load("Auta/full.xml")

    if out_path[-1] != '/':
        out_path += '/'
    files = utils.get_files(directory)
    for f in files:
        start_time = time.time()
        boundaries = []
        name_counter += 1
        if verbose > 0:
            print 'Detecting... ', os.path.basename(f)
        image = cv2.imread(f)
        # resized = rescaleImage(image, 300)
        # cv2.imshow("tmp", resized)
        # cv2.waitKey(0)
        resized = image
        detected = detect_multiscale(resized, hog_desc, svm, verbose=verbose)
        for obj in detected:
            # cv2.rectangle(image, obj.origin, obj.get_last(), (0,0,255), thickness=2)
            milis = int(round(time.time() * 1000))
            name = out_path + str(milis) + '.jpg'
            cv2.imwrite(name, obj.image)
            boundaries.append(obj.origin + obj.get_last())
        
        pick = non_max_suppression_fast(np.array(boundaries), 0.4)
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(resized, (startX, startY), (endX, endY), (0, 255, 0), 2)            
        cv2.imwrite(out_path + str(name_counter) + '.jpg', resized)
        print("--- %s seconds ---" % (time.time() - start_time))




##################
#
#   SOBEL
#
##################

def sobelY(img, y=12):
    blur = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

    ll = []
    r = img.shape[1] / 2
    
    for z in xrange(r - 10, r + 10):
        ll.append(sobely.transpose()[z])
    final = map(lambda x: np.mean(x), zip(*ll))

    print final
    mx = np.argmax(final[y:len(final)-y])
    mn = np.argmin(final[y:len(final)-y])
    if final[y+mx] <= 0:
        mx = mn
    if final[y+mn] >= 0:
        mn = mx
    
    return y + min(mx,mn), y + max(mx,mn)

def sobelX(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)

    ll = []
    r1 = 2 * img.shape[0] / 3
    r2 = 4 * img.shape[0] / 5
    final1 = sobelx[r1]
    final2 = sobelx[r2]
    final = map(np.mean, zip(*[final1, final2]))
    #print final, final1, final2
    
    mx = np.argmax(final)#, np.argmax(final2)
    mn = np.argmin(final)#, np.argmin(final2)
    if final[mx] <= 0:
        mx = mn
    if final[mn] >= 0:
        mn = mx
        
    return min(mn,mx), max(mn,mx)

def sobelY2(img, y=12):
    blur = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

    ll = []
    r = img.shape[1] / 2
    
    for z in xrange(r - 10, r + 10):
        ll.append(sobely.transpose()[z])
    final = map(lambda x: np.mean(x), zip(*ll))

    vals = []
    for i in range(len(final)-2):
        if final[i] < final[i + 1] > final[i + 2]:
            vals.append(i)
        elif final[i] > final[i + 1] < final[i + 2]:
            vals.append(i)
    #print final
    mx = np.argmax(final[y:len(final)-y])
    mn = np.argmin(final[y:len(final)-y])
    if final[y+mx] <= 0:
        mx = mn
    if final[y+mn] >= 0:
        mn = mx
    
    return vals#y + min(mx,mn), y + max(mx,mn)

def draw_rect(f, it, image, objects, obj, save=False):
    
    top = objects[obj]['top'][0]
    v = sobelY(top.image)
    y1 = top.origin[1] + int(round((v[0]+1)*obj.scale-1))

    bottom = objects[obj]['bottom'][0]
    v = sobelY2(top.image, y=4)
    y2 = bottom.origin[1] + int(round((v[-1]+1)*obj.scale-1))

    print v
    left = objects[obj]['left'][0]
    v = sobelX(left.image)
    #print '>>>', v, int(round((v[0]+1)*obj.scale-1)), obj.scale
    x1 = left.origin[0] + int(round((v[0]+1)*obj.scale-1))

    right = objects[obj]['right'][0]
    v = sobelX(right.image)
    #print v
    x2 = right.origin[0] + int(round((23+1)*obj.scale-1))
    #print '>>>', v, int(round((v[1]+1)*obj.scale-1)), obj.scale

    #print (x1, x2), (y1, y2)
    if x1 >= x2 or y1 >= y2:
        return

    print (x1,y1),(x2,y2)
    #t = prepare(f, it, image, (x1,y1), (x2,y2))
    t = (x1,y1), (x2,y2)
    print t
    if t is not None and save:
        save_img(image, t[0], t[1], (80,64), 'full', it, f)
        save_img(image, left.origin, left.get_last(), (24,48), 'left', it, f)
        save_img(image, right.origin, right.get_last(), (24,48), 'right', it, f)
        save_img(image, top.origin, top.get_last(), (64,32), 'top', it, f)
        save_img(image, bottom.origin, bottom.get_last(), (64,32), 'bottom', it, f)
    
    #cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0))
    print 20 * '-'
    return t

def prepare(f, it, img, p1, p2):
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

    #print (x0, x1), (y0, y1)
    if y0 >= 0 and y1 < img.shape[0]:
        return ((x0,y0), (x1,y1))
    return None
    
def rescale(p, win, scale):
    x = p[0] + int(round(win[0]*obj.scale-1))
    y = p[1] + int(round(win[1]*obj.scale-1))
    return (x,y)

def rescaleImage(image, height, width=None):
    r = float(height) / image.shape[0]
    dim = (int(image.shape[1] * r), height)
    print(height)
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def save_img(image, p1, p2, size, name, it, f):
    img = image[p1[1]:p2[1]+1, p1[0]:p2[0]+1].copy()
    if img is not None:
        img = cv2.resize(img, size)
        milis = int(round(time.time() * 1000))
        name = os.path.dirname(f) + '/Out/Done/' + name + '/' + str(it) + '_' + str(milis) + '.jpg'
        cv2.imwrite(name, img)





