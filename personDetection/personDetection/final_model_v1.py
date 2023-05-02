
# LIBRERIAS
from cmath import inf
from tabnanny import check
from time import ctime
import cv2 , imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import time
import argparse
# Histogram of Oriented Gradientes 

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Funcion principal
cap = cv2.VideoCapture(1)#Captura de video

def main():
    
    fontOne = cv2.FONT_HERSHEY_SIMPLEX #Declaramos tipo texto
    cTime = 0 
    pTime = 0
    #tolerancia de cordenadas 
    tolerance = 15
    #iniciamos el conteo de personas
    person = 0 
    past_cordanate= inf
    # person_count
    while True:
        # Conseguimos los fps
        
        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime
        key = cv2.waitKey(1)
        # Leemos el video o la camacar
        check,frame = cap.read()
        # Ajustamos para poder mejorar la calidad
        frame = imutils.resize(frame,width=min(400, frame.shape[1]))#hacemos rezise de la imagen
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Convertimos en gray la imgen
        bounding_box_cordinates,weights = HOGCV.detectMultiScale(frame_gray,winStride = (4,4),padding = (8,8),scale = 1.02) # Decalaramos variables de ajuste
        pick = non_max_suppression(bounding_box_cordinates,probs = None,overlapThresh=0.5) # quitamos el overlaping
        
        #iniciamos las personas detectadas
        for i,(x,y,w,h) in enumerate(pick):
            # if weights[i] < 0.13:
            #     not_certify+=1
            # elif weights[i] < 0.3 and weights[i] > 0.13:
            #     not_certify+=1 
            
            # Logica de que entre una personas
            if (past_cordanate == x or abs(past_cordanate-x)<=tolerance):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #En caso de que un pelado salga por la entrada
            elif x <= 10:
                if past_cordanate >x :
                    person-=1
                elif weights[i] < 0.7 and weights[i] > 0.3:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
                    print(f'x:{x},y:{y}')
                    person+=1
                elif weights[i] > 0.7:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    print(f'x:{x},y:{y}')
                    person+=1   
            else :
                if weights[i] < 0.7 and weights[i] > 0.3:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 122, 255), 2)
                    print(f'x:{x},y:{y}')
                    # person_stack.append(x)
                    # person_stack.pop(1)
                    
                elif weights[i] > 0.7:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    print(f'x:{x},y:{y}')
                    # person_stack.pop(1)
            past_cordanate = x 
                
        # frame_people_count=len(pick)
        # total_people_count+=person        
        cv2.putText(frame,f'{person}',(300,50),fontOne,0.8,(255,0,0),2)
        cv2.putText(frame,f'fps: {fps}',(30,50),fontOne,0.8,(255,0,0),2)
        cv2.imshow('output',frame)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows
        
        
    


if __name__ == "__main__":
    main()