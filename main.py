import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys
from imutils import face_utils
import imutils
import dlib


def template_matching(original_image, template):
    original_array = np.array(original_image)
    template_array = np.array(template)

    if template_array.shape[0] % 2 == 0:
        template_array = np.delete(template_array, 1, 0)

    if template_array.shape[1] % 2 == 0:
        template_array = np.delete(template_array, 1, 1)

    radius_h = template_array.shape[0] // 2
    radius_v = template_array.shape[1] // 2

    left_bound = radius_h + 1
    right_bound = original_array.shape[0] - radius_h

    upper_bound = radius_v + 1
    lower_bound = original_array.shape[1] - radius_v

    min_error = sys.maxsize
    res_coordinates = (0, 0)

    for i in range(left_bound, right_bound):
        for j in range(upper_bound, lower_bound):
            temp_slice = original_array[i - radius_h : i + radius_h + 1, j - radius_v : j + radius_v + 1]
            error = np.sum(np.power(temp_slice - template_array, 2))
            if error < min_error:
                min_error = error
                res_coordinates = i, j

    res_img = Image.fromarray(original_array[res_coordinates[0] - radius_h : res_coordinates[0] + radius_h,
                              res_coordinates[1]-radius_v : res_coordinates[1]+radius_v])
    res_img_2 =  cv.rectangle(original_array,
                        (res_coordinates[1] - radius_h, res_coordinates[0] + radius_v),
                        (res_coordinates[1] + radius_h, res_coordinates[0] - radius_v),
                        (0,255,0),3)
    return res_img_2


def viola_jones(original_image):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    img = np.asarray(original_image)[:,:,::-1].copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img

def face_symmetry(original_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor()
    





if __name__ == '__main__':
    image = Image.open('./cats.png')
    template = Image.open('./cat.png')
    # res = template_matching(image, template)
    res = viola_jones('./face.jpg')
    print(type(res))
    plt.imshow(res)
    plt.show()