import cv2
import os

def record_frames(video_path):
    os.makedirs("video_results/%s" % video_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print ('Read a new frame: ', success)
        if count %100 == 0:
            cv2.imwrite("video_results/%s/frame%d.jpg" % (video_path, count), image)
        count += 1

# record_frames('dolphin.mp4')
