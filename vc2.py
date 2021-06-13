import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')

algo = cv2.createBackgroundSubtractorMOG2() #For python3

def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

detect = []

count_line_position = 550
offset = 6
counter = 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    img_sub = algo.apply(blur)

    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilatada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('raw', dilatada)

    ret,dilatada = cv2.threshold(dilatada,127,255,cv2.THRESH_BINARY)

    # cv2.imshow('binary', dilatada)

    contour, h    = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, 550), (1200, 550), (255, 0, 0), 3)


    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w> 80) and (h>80)
        if not validate_contour:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        center = find_center(x, y, w, h)
        detect.append(center)

        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter +=1

            cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            detect.remove((x, y))
            cv2.line(frame, (25, 550), (1200, 550), (255, 255, 0), 3)
   
    cv2.imshow("output", frame)
    print(detect)

    if cv2.waitKey(15) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()