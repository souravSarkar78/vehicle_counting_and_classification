import cv2
import numpy as np


# Define the background substraction algo
algo = cv2.createBackgroundSubtractorMOG2()


# Define some variables 

# Tune these variables according to your input video

detection = []

allowed_height = 80
allowed_width = 80

line_position = 550

offset = 6
counter = 0

kernel = np.ones((5,5),np.uint8)


# Define the capture object
cap = cv2.VideoCapture('video.mp4')

# Function for finding the center of the detected object

def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

while True:
    ret, frame = cap.read()

    frame_height, frame_width, c = frame.shape

    # Perform background substraction technique to detect vahicles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    img_sub = algo.apply(blur)

    cv2.imshow("Foreground", img_sub)
    # Filter substracted output

    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    mask = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ret,binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    # Find the contour of the final mask

    contour, h    = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the count line 

    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 3)



    for (i, c) in enumerate(contour):

        # Finf rectangle boundaries of the detected contours

        (x, y, w, h) = cv2.boundingRect(c)

        # Valiadating the rectangles
        validate_contour = (w> allowed_width) and (h>allowed_height)
        if not validate_contour:
            continue

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Find the center of the rectangle for detection

        center = find_center(x, y, w, h)
        detection.append(center)

        # Draw circle in the middle of the rectangle
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (p, q) in detection:
            if q <(line_position+offset) and q >(line_position-offset):

                cv2.line(frame, (0, 550), (frame_width, 550), (0, 255, 0), 3)
                counter +=1

            
            # print(detection)
   
            detection.remove((p, q))

    # Show the vehicle counting on the frame
        
    cv2.putText(frame, "Vehicle: "+str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("output", frame)
    

    if cv2.waitKey(15) == ord('q'):
        break

# Releasing the capture object and finally destroy all the active windows
cap.release()
cv2.destroyAllWindows()