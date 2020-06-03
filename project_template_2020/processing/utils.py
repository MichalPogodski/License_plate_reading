import numpy as np
import cv2 as cv

MIN_MATCH_COUNT = 10

#def find_test():
    # img1 = cv.imread('PL.png', 0)  # queryImage
    # img2 = cv.imread('toFind.jpg', 0)  # trainImage
    # # Initiate SIFT detector
    # sift = cv.SIFT_create()
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    # if len(good) > MIN_MATCH_COUNT:
    #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #     matchesMask = mask.ravel().tolist()
    #     h, w, d = img1.shape
    #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #     dst = cv.perspectiveTransform(pts, M)
    #     img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    # else:
    #     print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    #     matchesMask = None
    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # cv.imshow('window', img3)
    #plt.imshow(img3, 'gray'), plt.show()

def thres(image):
    image = cv.resize(image, (600, 600))
    imgToTresh = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, img_threshed = cv.threshold(imgToTresh, 110, 255, cv.THRESH_BINARY_INV)
    kernel_cl = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(img_threshed, cv.MORPH_CLOSE, kernel_cl)
    kernel_op = np.ones((2, 2), np.uint8)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_op)
    kernel_er = np.ones((2, 2), np.uint8)
    erosion = cv.erode(opening, kernel_er, iterations=1)

    cv.imshow('window', erosion)
    while True:
        key_code = cv.waitKey(10)
        if key_code == 27:
            break
    cv.destroyAllWindows()


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    thres(image)



    return 'PO12345'
