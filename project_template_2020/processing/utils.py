import numpy as np
import cv2 as cv
import imutils
import os
from skimage import measure


def contour_to_rect(image):

    img = cv.resize(image, (620, 480))
    to_ret = img.copy()
    #start with preproc. for signidicant contours recognition (!!!red cooper!!!)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.bilateralFilter(gray, 11, 27, 27)
    edged = cv.Canny(blurred, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    cnts = imutils.grab_contours([contours, hierarchy])
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    screenCnt = None

    #looking for conoutrs of license plate (!!!red cooper!!!) DIFFERENT SOLUTION FOR BBX DETECT!!!!!!!!!
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv.drawContours(img, [screenCnt], 0, (0, 255, 0), 1)

    #######################################################################################################################
    # contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    # for cnt in contours:
    #     x, y, w, h = cv.boundingRect(cnt)
    #     ex = False
    #     if 4.0 < w / h < 6.0 and w >= 200:
    #         p0, p1, p2, p3 = (x,y), (x+w, y), (x, y+h), (x+2, y+h)
    #         print(p0, p1, p2, p3)
    #         cv.imshow('ts', img)
    ##############################################################################################
    p0, p2, p3, p1 = screenCnt[0][0], screenCnt[1][0], screenCnt[2][0], screenCnt[3][0]
    corn = []
    dict = {}
    dict[sum(p0)] = p0
    dict[sum(p1)] = p1
    dict[sum(p2)] = p2
    dict[sum(p3)] = p3
    srt_crn = sorted(dict)

    for idx in range(4):
        corn.append(dict[srt_crn[idx]])


    p0 = corn[0]
    p1 = corn[2]
    p2 = corn[1]
    p3 = corn[3]
    # p0 = [p0[0] - 5, p0[1] - 5]
    # p1 = [p1[0] + 5, p1[1] - 5]
    # p2 = [p2[0] - 5, p2[1] + 5]
    # p3 = [p3[0] + 5, p3[1] + 5]

    rect = np.array([p0, p1, p2, p3], np.float32)
    dst = np.array([[0, 0], [600, 0], [0, 150], [600, 150]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (600, 150))
    warped_clean = cv.warpPerspective(to_ret, M, (600, 150))

    return warped_clean



def thresh_chars(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.bilateralFilter(gray, 101, 17, 17)
    img_threshed = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 1)
    #ret, img_threshed = cv.threshold(blurred, 130, 255, cv.THRESH_BINARY_INV)
    kernel_er = np.ones((3, 3), np.uint8)
    erosion = cv.erode(img_threshed, kernel_er, iterations=1)
    # kernel_op = np.ones((3, 3), np.uint8)
    # opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel_op)
    # kernel_di = np.ones((3, 3), np.uint8)
    # dilation = cv.dilate(opening, kernel_di, iterations=1)
    return erosion

def segment(img, clr):
    edged = cv.Canny(img, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    characters = {}
    segmented = {}
    to_ex = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        ex = False
        if 0.4 < h/w < 4.0 and h >= 100:
            for elem in to_ex:
                if x <= elem <= (x+w):
                    ex = True
                    break
            if not ex:
                cp =img.copy()
                ch = cp[y:y+h, x:x+w]
                characters[x] = ch
                to_ex.append(x+w/2)

        sort = sorted(characters)

    print(len(characters))
    # cv.imshow('tst', img)
    for idx, key in enumerate(sort):
        segmented[idx] = characters[key]
        # cv.imshow(str(idx), segmented[idx])

    return segmented

def compare(detected, pattern):
    det = cv.resize(detected, (60, 120))
    pat = cv.resize(pattern, (60, 120))
    contours, hierarchy = cv.findContours(det, 2, 1)
    cnt1 = contours[0]
    contours, hierarchy = cv.findContours(pat, 2, 1)
    cnt2 = contours[0]
    ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
    s = measure.compare_ssim(det, pat)
    #return ret
    return s


def perf_comparison(segmented):
    recognition = []

    for e in segmented:
        elem = segmented[e]
        val = {}
        imgs = {}
        for filename in os.listdir('/home/michal/RiSA/SW/Number_plate_reading/symb'):
            ind = filename.find('.')
            name = str(filename[0:ind])
            img = cv.imread(os.path.join('/home/michal/RiSA/SW/Number_plate_reading/symb', filename))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret = compare(elem, gray)
            val[ret] = name
            imgs[ret] = gray
        srt_val = sorted(val)
        vals = list(srt_val)
        idx = len(vals)
        recognition.append(val[vals[idx - 1]])
        #pom = cv.imread(os.path.join('/home/michal/RiSA/SW/Number_plate_reading/symb', 'P.png'))
        #pom2 = cv.cvtColor(pom, cv.COLOR_BGR2GRAY)
        #print(compare(elem, pom2))
        #cv.imshow('det_'+str(vals[0]), imgs[vals[0]])
    print(recognition)
    return recognition


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    contour_to_rect(image)
    plate = contour_to_rect(image)
    threshed = thresh_chars(plate)
    #cv.imshow('test', threshed)
    segmented = segment(threshed, plate)
    recognition = perf_comparison(segmented)





    cv.waitKey()

    return 'PO12345'
