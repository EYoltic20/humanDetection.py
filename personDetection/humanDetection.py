from glob import glob
import cv2 , imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
# Histogram of Oriented Gradientes 
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# DETECT
global person 
# Valores para scale 
# 0.25 - cerca
#0.5 -media
#1- noram 1 a 1

def detect(frame,p,temp):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fontOne = cv2.FONT_HERSHEY_SIMPLEX
    bounding_box_cordinates,weights = HOGCV.detectMultiScale(frame_gray,winStride = (4,4),padding = (8,8),scale = 0.50)
    bounding_box_cordinates = np.array([[x,y,x+w,y+h] for (x,y,w,h) in bounding_box_cordinates])
    pick = non_max_suppression(bounding_box_cordinates,probs = None,overlapThresh=0.65)
    person = p
    temp_postion = temp
    for x,y,w,h in pick:
        cv2.rectangle(frame,(x,y),(w,h),(0,255,0),2)
        cv2.putText(frame,f'person{person}',(x,y),fontOne,0.5,(0,0,255),1)
        if ((x,y,w,h) in temp_postion):
            # temp_postion.pop()
            continue
        else:
            person +=1
            temp_postion.append((x,y,w,h))    
        cv2.putText(frame,'Status:Detection',(40,40),fontOne,0.8,(255,0,0),2)
    cv2.putText(frame,f'Total Persons: {person}',(40,70),fontOne,0.8,(255,0,0),2)
    cv2.imshow('output',frame)
    return frame,person,temp_postion


def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    while True:
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    # elif video_path is not None:
    #     print('[INFO] Opening Video from path.')
    #     detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])
        
def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    image = imutils.resize(image, width = min(800, image.shape[1])) 
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite('/Users/emilioymartinez/Desktop/6_toSemestre/personDetection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')
    person = 0 
    temp = []
    while True:
        check, frame = video.read()
        frame,person,temp = detect(frame,person,temp)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args  
    
if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    args = argsParser()
    humanDetector(args)