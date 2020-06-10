import numpy as np
import cv2 as cv
import imutils

def contour_to_rect(image):

    img = cv.resize(image, (620, 480))
    to_ret = img.copy()
    #start with preproc. for signidicant contours recognition (!!!red cooper!!!)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.bilateralFilter(gray, 11, 29, 29)
    img_threshed = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    edged = cv.Canny(img_threshed, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours([contours, hierarchy])
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    screenCnt = None

    #looking for conoutrs of license plate (!!!red cooper!!!)
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv.drawContours(img, [screenCnt], 0, (0, 255, 0), 1)
    # cv.imshow('test', img)
    p0, p2, p3, p1 = screenCnt[0][0], screenCnt[1][0], screenCnt[2][0], screenCnt[3][0]

    rect = np.array([p0, p1, p2, p3], np.float32)
    dst = np.array([[0, 0], [600, 0], [0, 150], [600, 150]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (600, 150))
    warped_clean = cv.warpPerspective(to_ret, M, (600, 150))
    # cv.imshow("output", warped)
    # cv.imshow('test',  warped_clean)
    return warped_clean




def thresh_chars(img):
    #preproc of image, for characters recognition
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.bilateralFilter(gray, 101, 29, 29)
    ret, t1 = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    cv.imshow('1', t1)



def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    plate = contour_to_rect(image)
    thresh_chars(plate)
    cv.waitKey()

    return 'PO12345'
