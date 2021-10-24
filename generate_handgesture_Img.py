import re
import cv2
import os
import glob
import torch
from PIL import Image

cap = cv2.VideoCapture(1)
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

img_id = 0
gesture_name = 'test'
data_save_path = 'MyGestureDataset\\'+gesture_name


def get_latest_data(elem):
    img = elem.split('\\')[2]
    return int(re.sub('.png', '', img))


if os.path.exists(data_save_path):
    dataset_list = glob.glob(data_save_path+'\\*')
    print(dataset_list)
    if dataset_list:
        dataset_list.sort(key=get_latest_data)
        img_id = get_latest_data(dataset_list[-1])
    print('Dataset exists')
    print('size of exists:'+str(img_id))
    print('use keyboard ‘c’ to add new data images')
else:
    print('New Dataset')
    print('use keyboard ‘c’ to add new data images')
    os.mkdir(data_save_path)

while (True):
    ret, frame = cap.read()

    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) # 转换至YCrCb空间
    (y,cr,cb) = cv2.split(YCrCb) # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    # _, skin = cv2.threshold(cr1, 140, 255, cv2.THRESH_BINARY ) #直接二值化
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Ostu处理

    img_erosion = cv2.erode(skin, kernel_erosion)
    img_dilate = cv2.dilate(img_erosion, kernel_erosion)

    res = cv2.bitwise_and(frame, frame, mask=skin)

    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Max_area_index = 0
    for i in range(len(contours)):
        temp = cv2.contourArea(contours[i])
        if temp > cv2.contourArea(contours[i]):
            Max_area_index = i
    if 40000 < cv2.contourArea(contours[Max_area_index]) < 80000:
        # print(cv2.contourArea(contours[Max_area_index]))
        cv2.drawContours(res, contours[Max_area_index], -1, (0, 0, 255), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', skin)
    cv2.imshow('erosion', img_dilate)
    cv2.imshow('res', res)

    img_dilate64x64 = cv2.resize(img_dilate, (64, 64), cv2.INTER_LINEAR)
    cv2.imshow('img_dilate64x64', img_dilate64x64)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('c'):
        img_id += 1
        cv2.imwrite(data_save_path+'/'+str(img_id)+'.png', img_dilate64x64)
        print('save gesture of '+gesture_name+' images:'+str(img_id)+' success.')
    if k == ord('q'):
        break

cv2.destroyAllWindows()
