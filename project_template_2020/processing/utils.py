import numpy as np
import cv2 as cv
import imutils

def contour_to_rect(image):

    img = cv.resize(image, (620, 480))

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
    cv.drawContours(img, [screenCnt], 0, (0, 255, 0), 3)

    # mask, in case to find out, what are the real contours of detected license plate (!!!red cooper!!!)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv.bitwise_and(img, img, mask=mask)
    cv.imshow("mask", new_image)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))

    #get properly perspective (!!!red cooper!!! + !!!stackOF with grocery store!!!!)
    rect = np.array([[topy, topx], [bottomy, topx], [topy, bottomx], [bottomy, bottomx]], np.float32)
    dst = np.array([[0, 0], [597, 0], [0, 145], [597, 145]], np.float32)
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (600, 150))
    cv.imshow("output", warped)

    #tuning of new perspective (own idea)





    cv.waitKey()








def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    #thres(image)
    contour_to_rect(image)



    return 'PO12345'
