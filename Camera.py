import cv2
import threading
import numpy as np
from common import Sketcher
import os
from datetime import datetime
import os.path
import json
from twilio.rest import Client

directory = ""
timestamp = datetime.now()
# User Set Variables
# todo add ability for end users to set these variables
sensitivity = 500
resize = 30
time_thresh = 10
config_file_path = "C:\\Users\\pheig\\Documents\\GitHub\\security\\config_test.json"
photo = ""

def send_text_message(frame):
    print("Sending text message")
    print(photo)
    with open(config_file_path) as json_data_file:
        data = json.load(json_data_file)
        twilio = data['twilio']
        # /usr/bin/env python
        # Download the twilio-python library from twilio.com/docs/libraries/python
        from twilio.rest import Client

        # Find these values at https://twilio.com/user/account
        # To set up environmental variables, see http://twil.io/secure
        account_sid = twilio['account_sid']
        auth_token = twilio['auth_token']

        client = Client(account_sid, auth_token)

        client.api.account.messages.create(
            to=twilio['phone'],
            from_=twilio['twilio_phone'],
            media_url=[photo],
            body="Motion Detected")


def connection_string():
    with open(config_file_path) as json_data_file:
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

def motion(motion_count, frame, main_path):
    # If motion is for longer than a set time begin capturing
    global timestamp
    global directory
    if motion_count == time_thresh:  # 25 frames p/s for 3 seconds of motion
        now = datetime.now()
        delta = timestamp - now
        minute, second = divmod(delta.seconds, 60)
        # if it has been longer than 5 minutes since movement
        # create new folder
        # todo this feels messy need to cleanup logic
        print(second)
        if second > 5:
            # If it is not the first pass then generate video from images
            if directory != "":
                threading.Thread(target=generate_video, args=()).start()
                generate_video()
            dt_string = now.strftime("%d.%m.%Y.%H.%M.%S")
            directory = main_path + dt_string
            os.mkdir(directory)
            print(directory)
            timestamp = now
    if motion_count >= time_thresh:
        # todo make this more dynamic for a user
        path = directory + "\\frame" + str(motion_count) + ".jpg"
        print(path)
        global photo
        photo = path
        cv2.imwrite(path, frame)


# Video Generating function
def generate_video():
    image_folder = directory # make sure to use your folder
    video_name = 'capture.avi'
    os.chdir(image_folder)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if any
    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    # todo decide if end user should be able to pick fourcc format
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(video_name, fourcc, 60.0, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

# main video feed
def show_stream():
    print("show stream")
    connection = connection_string()
    capture = cv2.VideoCapture(connection)
    while True:
        ret, frame = capture.read()
        # Check for empty frame
        if frame.size == 0:
            print('Empty frame')
            continue
        frame = resize_frame(frame, resize)
        cv2.imshow("Main", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def main():
    print("I am starting from main")
    # Make empty frame objects using an empty list and np.array
    list = []
    gray2 = np.array(list)
    image_mask = np.array(list)
    # Establish connection to the webcam
    threading.Thread(target=show_stream, args=()).start()

    motion_count = 0
    main_path = "C:\\Users\\pheig\\Documents\\GitHub\\video\\"

    connection = connection_string()
    cap = cv2.VideoCapture(connection)
    while True:
        ret, frame = cap.read()
        # Check for empty frame
        if frame.size == 0:
            continue

        frame = resize_frame(frame, resize)

        # If mask is empty, set mask
        if image_mask.size == 0:
            image_mask = create_mask(frame)

        # If the resize parameter has changed after the mask was already set
        # apply the new resize to the mask
        if image_mask.shape[0:2] != frame.shape[0:2]:
            image_mask = cv2.resize(image_mask, (frame.shape[1], frame.shape[0]))

        # Convert frame to grayscale and apply gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)

        # If the initial frame is not set then set it
        if gray2.size == 0:
            gray2 = gray

        # Get the difference between the original frame
        # and the latest frame
        deltaframe = cv2.absdiff(gray, gray2)

        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None)
        # threshold = cv2.bitwise_and(threshold, threshold, mask=image_mask)

        # Draw rectangles around where motion is detected
        contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour:
            # increment frame count for motion capture
            motion_count += 1
        else:
            # set motion capture back to 0
            motion_count = 0

        motion(motion_count, frame, main_path)

        for i in contour:
            if cv2.contourArea(i) < sensitivity:
                continue

            (x, y, w, h) = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing", frame)
        # Process(target=cv2.imshow("Capturing", frame), args=()).start()
        #cv2.imshow("Threshold", threshold)
        # cv2.imshow("Mask", image_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_text_message(frame)
            threading.Thread(target=generate_video, args=()).start()
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()






