import numpy as np
import cv2 as cv
import imutils

def contour_to_rect(image):

    img = cv.resize(image, (620, 480))
    to_ret = img.copy()
    #start with preproc. for signidicant contours recognition (!!!red cooper!!!)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.bilateralFilter(gray, 11, 27, 27)
    edged = cv.Canny(blurred, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
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
    p0 = [p0[0] - 5, p0[1] - 5]
    p1 = [p1[0] + 5, p1[1] - 5]
    p2 = [p2[0] - 5, p2[1] + 5]
    p3 = [p3[0] + 5, p3[1] + 5]

    rect = np.array([p0, p1, p2, p3], np.float32)
    dst = np.array([[0, 0], [600, 0], [0, 150], [600, 150]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (600, 150))
    warped_clean = cv.warpPerspective(to_ret, M, (600, 150))

    return warped_clean



def thresh_chars(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.bilateralFilter(gray, 101, 37, 37)
    #img_threshed = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    ret, img_threshed = cv.threshold(blurred, 130, 255, cv.THRESH_BINARY_INV)
    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv.morphologyEx(img_threshed, cv.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv.erode(img_threshed, kernel, iterations=1)

    return img_threshed

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
    cv.imshow('tst', img)
    for idx, key in enumerate(sort):
        segmented[idx] = characters[key]
        cv.imshow(str(idx), segmented[idx])

    return segmented





def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    # contour_to_rect(image)
    plate = contour_to_rect(image)
    threshed = thresh_chars(plate)
    #cv.imshow('test', threshed)
    segmented = segment(threshed, plate)





    cv.waitKey()

    return 'PO12345'
