import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    flnames = []
    for filename in os.listdir(folder):
        flnames.append(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images, flnames

def preproc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 101, 57, 57)
    ret, img_threshed = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)
    #img_threshed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 1)
    kernel_op = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_op)
    kernel_cl = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cl)
    res = cv2.resize(img_threshed, (60, 120))

    return res


def main():
    images, flnames = load_images_from_folder('char')
    for idx, img in enumerate(images):
        img = preproc(img)
        cv2.imwrite('symb/'+str(flnames[idx]), img)
        print('symb/'+str(flnames[idx]))




if __name__ == '__main__':
    main()
