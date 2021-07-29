import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
time.sleep(3)

background = 0

for i in range(30):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = np.flip(frame, axis =1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (35,35), 0)

    lower = np.array([0,120,70])
    upper = np.array([10,255,255])
    mask01 = cv2.inRange(hsv, lower, upper)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])

    mask02 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask01+mask02 

    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    frame[np.where(mask==255)] = background[np.where(mask==255)]

    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('background', background)
    # cv2.imshow('mask01', mask01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()