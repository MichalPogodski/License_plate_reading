import numpy as np
import cv2 as cv
import os
from skimage import measure

not_det = 0

def find_plate(img):

    to_ret = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 3)
    _, thresh = cv.threshold(blurred, 145, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull = []
    for cnt in contours: hull.append(cv.convexHull(cnt))
    cnts = sorted(hull, key=cv.contourArea, reverse=True)[:10]
    box = None

    for i in range(len(cnts)):
        epsilon = 0.03 * cv.arcLength(cnts[i], True)
        apprx = cv.approxPolyDP(cnts[i], epsilon, True)
        if len(apprx) == 4:
            x, y, w, h = cv.boundingRect(apprx)
            if 2.5 <= w/h <= 7.5 and w >= img.shape[0]/3 and w < img.shape[0]:
                box = apprx
                break

    p0, p2, p3, p1 = box[0][0], box[1][0], box[2][0], box[3][0]
    corn = []
    dict = {}
    dict[sum(p0)] = p0
    dict[sum(p1)] = p1
    dict[sum(p2)] = p2
    dict[sum(p3)] = p3
    srt_crn = sorted(dict)

    for idx in range(4):
        corn.append(dict[srt_crn[idx]])

    p0 = [corn[0][0], corn[0][1] - 15]
    p1 = [corn[2][0], corn[2][1] - 15]
    p2 = [corn[1][0], corn[1][1] + 15]
    p3 = [corn[3][0], corn[3][1] + 15]

    rect = np.array([p0, p1, p2, p3], np.float32)
    dst = np.array([[0, 0], [600, 0], [0, 150], [600, 150]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped_clean = cv.warpPerspective(to_ret, M, (600, 150))

    return warped_clean


def adaptive_find_plate(img):
    to_ret = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 9)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 2)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull = []
    for cnt in contours: hull.append(cv.convexHull(cnt))
    cnts = sorted(hull, key=cv.contourArea, reverse=True)[:10]
    box = None

    for i in range(len(cnts)):
        epsilon = 0.03 * cv.arcLength(cnts[i], True)
        apprx = cv.approxPolyDP(cnts[i], epsilon, True)
        if len(apprx) == 4:
            x, y, w, h = cv.boundingRect(apprx)
            if 2.5 <= w / h <= 7.5 and w >= img.shape[0] / 3 and w < img.shape[0]:
                box = apprx
                break

    p0, p2, p3, p1 = box[0][0], box[1][0], box[2][0], box[3][0]
    corn = []
    dict = {}
    dict[sum(p0)] = p0
    dict[sum(p1)] = p1
    dict[sum(p2)] = p2
    dict[sum(p3)] = p3
    srt_crn = sorted(dict)

    for idx in range(4):
        corn.append(dict[srt_crn[idx]])

    p0 = [corn[0][0], corn[0][1] - 15]
    p1 = [corn[2][0], corn[2][1] - 15]
    p2 = [corn[1][0], corn[1][1] + 15]
    p3 = [corn[3][0], corn[3][1] + 15]

    rect = np.array([p0, p1, p2, p3], np.float32)
    dst = np.array([[0, 0], [600, 0], [0, 150], [600, 150]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped_clean = cv.warpPerspective(to_ret, M, (600, 150))

    return warped_clean


def thresh_chars(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    kernel_di = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(thresh, kernel_di, iterations=1)

    return dilation



def segment(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    characters = {}
    segmented = {}
    to_ex = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        ex = False
        if 0.4 < h/w < 4.0 and h >= 50:
            for elem in to_ex:
                if x <= elem <= (x+w):
                    ex = True
                    break
            if not ex:
                cp =img.copy()
                characters[x] = cp[y:y+h, x:x+w]
                to_ex.append(x+w/2)

        sort = sorted(characters)

    for idx, key in enumerate(sort):
        segmented[idx] = characters[key]

    return segmented


def compare(detected, pattern):
    det = cv.resize(detected, (60, 120))
    s = measure.compare_ssim(det, pattern)
    return s


def perf_comparison(segmented):
    recognition = ""
    banned = ['B', 'D', 'I', 'O', 'Z']
    banned_on_start = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for e in segmented:
        elem = segmented[e]
        val = {}
        for filename in os.listdir('processing/symb'):
            ind = filename.find('.')
            name = str(filename[0:ind])
            img = cv.imread(os.path.join('processing/symb', filename))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret = compare(elem, gray)
            val[ret] = name
        srt_val = sorted(val)
        vals = list(srt_val)
        idx = len(vals)

        if len(recognition) < 2:
            for i in range(len(banned_on_start)):
                symb = val[vals[idx - 1]]
                if symb == banned_on_start[i]:
                    idx -= 1
                    i = 0
        if len(recognition) >= 3:
            for i in range(len(banned)):
                symb = val[vals[idx - 1]]
                if symb == banned[i]:
                    idx -= 1
                    i = 0

        recognition += str(val[vals[idx - 1]])
    return recognition


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    global not_det

    try:
        plate = find_plate(image)
    except:

        try:
            plate = adaptive_find_plate(image)
        except:
            recognition = '???????'
            not_det += 1
        else:
            threshed = thresh_chars(plate)
            cv.imshow('detected plate_ADAPT******', threshed)

            try:
                segmented = segment(threshed)
                recognition = perf_comparison(segmented)
            except:
                recognition = '???????'

    else:
        threshed = thresh_chars(plate)
        cv.imshow('detected plate', threshed)

        try:
            segmented = segment(threshed)
            recognition = perf_comparison(segmented)
        except:
            recognition = '???????'



    if len(recognition) != 7:
        try:
            plate = adaptive_find_plate(image)
        except:
            recognition = '???????'
            not_det += 1

        else:
            threshed = thresh_chars(plate)
            cv.imshow('detected plate_ADAPT******', threshed)
            try:
                segmented = segment(threshed)
                recognition = perf_comparison(segmented)
            except:
                recognition = '???????'


    if len(recognition) < 7:
        for i in range(7 - len(recognition)):
            recognition += '?'
    elif len(recognition) > 7:
        rec = recognition
        recognition = rec[(len(rec)-7):]

    print('recognition ', recognition)
    cv.waitKey()
    print('not detected: ', not_det)

    return recognition
