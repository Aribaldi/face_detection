import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys
from imutils import face_utils
import imutils
import dlib
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS


def template_matching(original_img_path, template_path):

    original_img_path = Image.open(original_img_path)
    template_path = Image.open(template_path)

    original_array = np.array(original_img_path)
    template_array = np.array(template_path)

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

    # res_img = Image.fromarray(original_array[res_coordinates[0] - radius_h : res_coordinates[0] + radius_h,
    #                           res_coordinates[1]-radius_v : res_coordinates[1]+radius_v])
    res_img_2 =  cv.rectangle(original_array,
                        (res_coordinates[1] - radius_h, res_coordinates[0] + radius_v),
                        (res_coordinates[1] + radius_h, res_coordinates[0] - radius_v),
                        (0,255,0),3)
    return Image.fromarray(res_img_2)


def viola_jones(img_path):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return Image.fromarray(img)


def face_symmetry(img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    img = cv.imread(img_path)
    #img = imutils.resize(img, width=500)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for i, rect in enumerate(rects):
        shape = predictor(gray, rect) #nose: shape[8] shape[27]
        shape = face_utils.shape_to_np(shape)
        l_start, l_end = FACIAL_LANDMARKS_68_IDXS['left_eye']
        r_start, r_end = FACIAL_LANDMARKS_68_IDXS['right_eye']
        lefteyepts = shape[l_start:l_end]
        righteyepts = shape[r_start:r_end]
        lefteyecenter_x = tuple(lefteyepts.mean(axis=0).astype('int'))[0]
        righteyecenter_x = tuple(righteyepts.mean(axis=0).astype('int'))[0]
        top_nose_x = shape[8][0]
        bottom_nose_x = shape[27][0]
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        nose_top = (top_nose_x, y + h)
        nose_bottom = (bottom_nose_x, y)
        re_top = (righteyecenter_x, y + h)
        re_bottom = (righteyecenter_x, y)

        le_top = (lefteyecenter_x, y + h)
        le_bottom = (lefteyecenter_x, y)
        cv.line(img, nose_top, nose_bottom, (255, 0, 0), 2)
        cv.line(img, re_top, re_bottom , (255, 0, 0), 2)
        cv.line(img, le_top, le_bottom, (255, 0, 0), 2)
    return Image.fromarray(img)


if __name__ == '__main__':
    res = template_matching('./cats.png', './cat.png')
    #res = viola_jones('./face.jpg')
    #res = face_symmetry('./face.jpg')
    plt.imshow(res)
    plt.show()