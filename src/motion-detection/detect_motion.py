# Import Libraries
import numpy as np
import cv2
# import matplotlib.pyplot as plt

# Data path and settings
DATA_DIR = "../../data/basic-videos/"
VIDEO_STREAM = DATA_DIR + "cars.mp4" 

# Get video stream
video = cv2.VideoCapture(VIDEO_STREAM) # 0 => default web cam. 1 => external web cam. or file name (string)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20
sz = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Open and set props
vout = cv2.VideoWriter()
vout.open('output.mp4',fourcc,fps,sz,True)

# Get 2 frames
ret1, frame1 = video.read()
ret2, frame2 = video.read()

while True:
    # Convert to grayscale first
    # Move these functions to utils
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur to remove noise and smoothen image
    frame1_blur = cv2.GaussianBlur(frame1, (21, 21), 0)
    frame2_blur = cv2.GaussianBlur(frame2, (21, 21), 0)

    # Get difference between 2 frames
    diff = cv2.absdiff(frame1_blur, frame2_blur)

    # Define threshold
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1] # returns [threshold value, image converted]
    
    # Now calculate the pixels (white) if greater than some threshold then there is a motion.
    white_pixels = np.sum(thresh) / 255 # its either 0 or 255
    
    # Total pixels
    rows, cols, _ = thresh.shape
    total_pixels = rows * cols
    
    # Now condition.
    # If white_pixels are 1% of total pixels then motion detected
    if white_pixels > 0.01 * total_pixels:
        # Motion detected. Do something
        cv2.putText(frame1, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
    vout.write(frame1)
    # Show the frame
    cv2.imshow("Motion", frame1)
    
    # Now replace frame
    frame1 = frame2
    ret, frame2 = video.read()
    
    if not ret:
        break
        
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
        
vout.release()
cv2.destroyAllWindows()