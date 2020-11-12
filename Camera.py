import cv2
import numpy as np
from common import Sketcher
import os
from datetime import datetime
import os.path
import json


def connection_string():
    with open("config_test.json") as json_data_file:
        data = json.load(json_data_file)
        cam = data['camera']
        string = cam['url_pre'] + cam['user'] + ':' + cam['password'] + cam['url_post'] + ':' + cam['port']
        return string

def create_mask(image):
    print('Create mask function called')
    # Create an image for sketching the mask
    # If mask was created before then import it as grayscale
    if os.path.isfile('mask.png'):
        image_mask = cv2.imread('mask.png', 0)
        image_mask = cv2.bitwise_not(image_mask)
        return image_mask
    else:
        image_mask = image.copy()
        sketch = Sketcher('Image', [image_mask], lambda : ((255, 255, 255), 255))
 
        # Sketch a mask
        # todo - add a window with instruction for the end user
        while True:
            ch = cv2.waitKey()
            if ch == 27: # ESC - exit
                break
            if ch == ord('r'): # r - mask the image
                break
            if ch == ord(' '): # SPACE - reset the inpainting mask
                image_mask[:] = image
                sketch.show()

        # define range of white color in HSV
        lower_white = np.array([0,0,255])
        upper_white = np.array([255,255,255])

        # Create the mask
        image_mask = cv2.inRange(image_mask, lower_white, upper_white)
        cv2.imwrite('mask.png', image_mask)

        return image_mask

def resize_frame(frame, resize):
    # Resize the image by given percentage
    width = int(frame.shape[1] * (resize / 100.0))
    height = int(frame.shape[0] * (resize / 100.0))
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    return frame

# Make empty frame objects using an empty list and np.array
list = []
gray2 = np.array(list)
image_mask = np.array(list)

motion_count = 0
directory = ""

# Establish connection to the webcam
connection = connection_string()
cap = cv2.VideoCapture(connection)

# User Set Variables
# todo add ability for end users to set these variables
sensitivity = 200
resize = 30

while True:
    ret, frame = cap.read()
    # Check for empty frame
    if frame.size == 0:
        print('Empty frame')
        continue

    frame = resize_frame(frame, resize)
    
    # If mask is empty, set mask
    if image_mask.size == 0:
        image_mask = create_mask(frame)

    # If the resize parameter has changed after the mask was already set
    # apply the new resize to the mask
    if image_mask.shape[0:2] != frame.shape[0:2]:
        image_mask = cv2.resize(image_mask,(frame.shape[1], frame.shape[0]))


    # Convert frame to grayscale and apply 
    # gaussian blur for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    # If the initial frame is not set set it
    if gray2.size == 0:
        gray2 = gray
    
    # Compare frames
    deltaframe = cv2.absdiff(gray, gray2)
    
    threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None)
    #threshold = cv2.bitwise_and(threshold, threshold, mask=image_mask)

    # Draw rectangles around where motion is detected
    contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contour:
        # increment frame count for motion capture
        motion_count += 1
    else:
        # set motion capture back to 0
        motion_count = 0

    print(motion_count)

    # If motion is for longer than 10 frames begin capturing
    # todo use timestamp to determine duration
    if motion_count == 10:
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y.%H.%M.%S")
        directory = dt_string
        os.mkdir(directory)
        print(directory)
    if motion_count >= 10:
        # todo make this more dynamic for a user
        main_path = "C:\\Users\\pheig\\Documents\\GitHub\\video\\"
        path = main_path + directory + "\\frame" + str(motion_count) + ".png"
        print(path)
        cv2.imwrite(path, frame)

    for i in contour:
        # todo - make this a variable that users can set for sensitivity to motion
        if cv2.contourArea(i) < sensitivity:
            continue

        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow("Capturing",frame)
    #cv2.imshow("Threshold", threshold)
    #cv2.imshow("Mask", image_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





