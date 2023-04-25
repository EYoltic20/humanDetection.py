from time import ctime
import cv2 , imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import time
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

def detect(frame):
    
    fontOne = cv2.FONT_HERSHEY_SIMPLEX
    frame = imutils.resize(frame,width=min(400, frame.shape[1]))
    # if frame.shape[1] < 400: # if image width < 400
    #     (height, width) = image.shape[:2]
    #     ratio = width / float(width) # find the width to height ratio
    #     image = cv2.resize(frame, (400, width*ratio))
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_box_cordinates,weights = HOGCV.detectMultiScale(frame_gray,winStride = (4,4),padding = (8,8),scale = 1.02)
    # bounding_box_cordinates = np.array([[x,y,x+w,y+h] for (x,y,w,h) in bounding_box_cordinates])
    pick = non_max_suppression(bounding_box_cordinates,probs = None,overlapThresh=0.5)
    person = 0
    # temp_postion = temp
    for i,(x,y,w,h) in enumerate(pick):
        
        # if weights[i] < 0.13:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # elif weights[i] < 0.3 and weights[i] > 0.13:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if weights[i] < 0.7 and weights[i] > 0.3:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
            person+=1
            
        if weights[i] > 0.7:
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            person +=1
        
            # if ((x,y,w,h) in temp_postion):
            #      # temp_postion.pop()
            #      continue
            # else:
            
                # temp_postion.append((x,y,w,h))
         
    cv2.putText(frame,"HIGH Confidence",(10, 25),fontOne,0.8,(0, 255, 0),2)
    cv2.putText(frame,"Moderate confidences",(10, 55),fontOne,0.8,(50, 122, 255),2)
    cv2.putText(frame,"Low Confidence",(10, 85),fontOne,0.8,(0, 0, 255),2)
        # cv2.putText(frame,'Status:Detection',(40,40),fontOne,0.8,(255,0,0),2)
    cv2.putText(frame,f'Total Persons: {person}',(10,105),fontOne,0.8,(255,0,0),2)
    cv2.imshow('output',frame)
    return frame

    
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
    image = imutils.resize(image, width = min(1000, image.shape[1])) 
    result_image = detect(image)
    if output_path is not None:
        cv2.imwrite('/Users/emilioymartinez/Desktop/6_toSemestre/personDetection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detectByCamera(writer):   
    video = cv2.VideoCapture(1)
    fontOne = cv2.FONT_HERSHEY_SIMPLEX
    print('Detecting people...')
    cTime = 0 
    pTime = 0
    while True:
        cTime = time.time()
        fps = 1/(cTime - pTime)
        check, frame = video.read()
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        cv2.putText(frame,"fps:{fps}",(50,105),fontOne,0.8,(0, 255, 0),2)
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