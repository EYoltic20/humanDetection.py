from ctypes import resize
import numpy as np
import cv2
from sklearn.svm import LinearSVC

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Casos para entrenar
positive  = ['/Users/emilioymartinez/Desktop/6_toSemestre/personDetection/semana_tec.jpeg']
negative =['/Users/emilioymartinez/Desktop/6_toSemestre/personDetection/personDetection/trained/samples/space.jpg']

win_size = (64,128)




# positive
pos_feature = []
for file in positive:
    img = cv2.imread(file)
    img =cv2.resize(img,win_size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = hog.compute(img_gray)
    pos_feature.append(hist)
pos_feature = np.array(pos_feature,dtype=np.float32)
pos_labels = np.ones((pos_feature.shape[0],1),dtype=np.float32)

    
# NEGATIVE
neg_feature = []
for file in negative:
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = hog.compute(img_gray)
    neg_feature.append(hist)
neg_feature = np.array(neg_feature,dtype=np.float32)
neg_labels = np.zeros((pos_feature.shape[0],1),dtype=np.float32)

print(len(pos_feature))
print(len(neg_feature))
neg_feature = neg_feature.reshape(-1, 3780)
samples = np.concatenate((pos_feature,neg_feature),axis = 0)
labels = np.concatenate((pos_labels,neg_labels),axis = 0)



# Hacer random 
suffle = np.random.permutation(samples.shape[0])
samples = samples[suffle]
labels = samples[suffle]


# Entrenarlo usando transfer learning
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_EPS, 100, 1e-6))
# train_data = cv2.ml.TrainData_create(samples, cv2.ml.ROW_SAMPLE, labels, varType=cv2.ml.VAR_CATEGORICAL)

svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

# PONER EL DETETCTOR COMO ENTRENADDO



svm.save('modeloEntrenado1.xml')
# hog.setSVMDetector(svm.getSupportVector()[0]*svm.getDecisionFunction(0)[0])


