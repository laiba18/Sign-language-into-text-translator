import cv2
import numpy as np
import SVM as st


# Get the biggest Controur
def getMaxContour(contours, minArea=200):
    maxC = None
    maxArea = minArea
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > maxArea):
            maxArea = area
            maxC = cnt
    return maxC


# Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,svm):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    imgT = img[y:y + h, x:x + w]
    imgT = cv2.bitwise_and(imgT, imgT, mask=th1[y:y + h, x:x + w])
    imgT = cv2.resize(imgT, (200,200))
    imgTG = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    resp = st.predict(svm,imgTG)
    #cv2.imshow('imgTG',imgTG)
    if resp==1:
        label=1
    elif resp==2:
        label=2
    elif resp==3:
        label=3
    else:
        label=4
    return label