import numpy as np
import cv2
import SVM as st
import UTIL as ut

model=st.train_svm(4)
#font = cv2.FONT_HERSHEY_SIMPLEX
temp=0
previouslabel=None
previousText=" "
label = None
cam = cv2.VideoCapture(0)
#font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, img = cam.read()
    cv2.rectangle(img, (0, 70), (319, 950), (0, 255, 0), 3)
    img1 = img[70:950, 0:319]
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    lower_range = np.array([0, 10, 60])
    upper_range = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # blur = cv2.blur(mask, (5, 5))
    blurred_hsv = cv2.GaussianBlur(mask, (9, 9), 0)
    res = cv2.bitwise_and(img1, img1, mask=blurred_hsv)
    cv2.imshow("frame",img)
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = ut.getMaxContour(contours, 4000)
    #cv2.putText(img1, "hello world!", (120, 90), font, 1, (200, 255, 255), 2)

    if cnt is not None:
          label=ut.getGestureImg(cnt,img,mask,model)
          if (label is not None):
                if (temp == 0):
                    previouslabel = label
                if previouslabel == label:
                    previouslabel = label
                    temp += 1
                else:
                   temp = 0
                #print(label)
                print(chr(label+64))
                #text=ch.getText(text=chr(label+64))
                #cv2.putText(img, label, (0,90), font, 1, (200, 0, 0), 3)
                #draw = ImageDraw.Draw(label)
                #font = ImageFont.truetype("sans-serif.ttf", 16)
                #draw.text((0, 0), label, (255, 255, 255), font=font)

    else:
        print(None)
        continue

    #cv2.putText(img, label, (50, 150), font, 8, (0, 125, 155), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cam.release()
cv2.destroyAllWindows()

