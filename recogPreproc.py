import cv2
import os

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
    blurred = cv2.bilateralFilter(gray, 101, 37, 37)
    ret, img_threshed = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV)

    return img_threshed


def main():
    images, flnames = load_images_from_folder('char')
    for idx, img in enumerate(images):
        img = preproc(img)
        cv2.imwrite('symb/'+str(flnames[idx]), img)
        print('symb/'+str(flnames[idx]))




if __name__ == '__main__':
    main()
