import cv2


video_path = 'video.mp4'

# capture video from camera
cap = cv2.VideoCapture(0)
# use cap.isOpened() to check whether the capture is initialized, or cap.open() to initialize

# check the width and height
print(cap.get(3)) # width
print(cap.get(4)) # height
# modify a 640*480 image to 320*240
ret = cap.set(3, 320) # width
ret = cap.set(4, 240) # height

while True:
    # capture frame-by-frame
    # ret is a boolean value
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# play video from file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save a video
cap = cv2.VideoCapture(0)

# define the codec and create a video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps=20.0, frameSize=(640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

