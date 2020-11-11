import cv2
import numpy as np
from common import Sketcher
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
        cv2.imshow('Mask', image_mask)
        return image_mask
    else:
        image_mask = image.copy()
        sketch = Sketcher('Image', [image_mask], lambda : ((255, 255, 255), 255))
 
        # Sketch a mask
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


# Make empty frame objects using an empty list and np.array
list = []
gray2 = np.array(list)
image_mask = np.array(list)

# Establish connection to the webcam
connection = connection_string()
cap = cv2.VideoCapture(connection)

while True:
    ret, frame = cap.read()
    # Check for empty frame
    if frame.size == 0:
        print('Empty frame')
        continue

    # Resize the image by 33%
    width = int(frame.shape[1] * .33)
    height = int(frame.shape[0] * .33)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    
    # If mask is empty, set mask
    if image_mask.size == 0:
        image_mask = create_mask(frame)
    
    # Convert frame to grayscale and apply 
    # gaussian blur for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    
    if gray2.size == 0:
        gray2 = gray
    
    # Compare frames
    deltaframe = cv2.absdiff(gray, gray2)
    
    threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None)

    threshold = cv2.bitwise_and(threshold, threshold, mask=image_mask)
    
    contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in contour:
        if cv2.contourArea(i) < 200:
            continue

        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow("Capturing",frame)
    cv2.imshow("Threshold", threshold)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





