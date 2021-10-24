import cv2
import torch

cap = cv2.VideoCapture(1)
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

font = cv2.FONT_HERSHEY_SIMPLEX

mode = torch.load('Epochs\\Gesture_recongize\\2021_10_22 23_11_10epoch21.pth')
device = torch.device('cuda:0')
mode.to(device)
while (True):
    ret, frame = cap.read()

    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    # _, skin = cv2.threshold(cr1, 140, 255, cv2.THRESH_BINARY ) #直接二值化
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理

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

    inputs = torch.from_numpy(img_dilate64x64).reshape([-1, 1, 64, 64])
    inputs = inputs.float()
    outputs = mode(inputs.to(device))
    _, pred = torch.max(outputs.data, 1)

    label = ''
    if pred.item() == 0:
        label = 'paper'
    elif pred.item() == 1:
        label = 'rock'
    elif pred.item() == 2:
        label = 'thumb'
    elif pred.item() == 3:
        label = 'up'
    elif pred.item() == 4:
        label = 'V'

    img_dilate64x64 = cv2.putText(img_dilate64x64, label, (10, 10), font, 0.5, (255, 255, 255), 2)
    cv2.imshow('img_dilate64x64', img_dilate64x64)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
