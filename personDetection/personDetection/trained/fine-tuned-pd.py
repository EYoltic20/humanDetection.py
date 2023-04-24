import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Casos para entrenar
positive  = ["/Users/emilioymartinez/Desktop/6_toSemestre/personDetection/semana_tec.jpeg"]
negative =["/Users/emilioymartinez/Desktop/6_toSemestre/personDetection/personDetection/trained/samples/space.jpg"]

samples = []
labels = []


# positive
for file in positive:
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hogT = cv2.HOGDescriptor()
    hist = hogT.compute(img_gray)
    samples.append(hist)
    labels.append(1)
# NEGATIVE
for file in negative:
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hogT = cv2.HOGDescriptor()
    hist = hogT.compute(img_gray)
    samples.append(hist)
    labels.append(1)
    
    



# Entrenarlo usando transfer learning
svm  = cv2.ml.SVM.create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,1000,1e-3))
# Convertir a np
samples = np.matrix(samples)
labels = np.array(labels)

# entrenar el svm 
svm.train(samples,cv2.ml.ROW_SAMPLE,labels) 
# PONER EL DETETCTOR COMO ENTRENADDO

svm.save('modeloEntrenado1.xml')
# hog.setSVMDetector(svm.getSupportVector()[0]*svm.getDecisionFunction(0)[0])


