import numpy as np
import cv2


video_path = '/home/charlie/Music/CloudMusic/MV/Ingrid_Michaelson-Afterlife.mp4'

# capture video from camera
cap = cv2.VideoCapture(0)
# use cap.isOpened() to check whether the capture is initialized
# and use cap.open() to initialize

# check the width and height
print(cap.get(3)) # width
print(cap.get(4)) # height
# modify a 640*480 image to 320*240
ret = cap.set(3, 320) # width
ret = cap.set(4, 240) # height

while True:
    # capture frame-by-frame
    # read() returns a bool value
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# playing video from file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# saving a video
cap = cv2.VideoCapture(0)

# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
        else:
            break

# release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
