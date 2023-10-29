import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

import os
import glob
def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

 
cap = cv2.VideoCapture(0)

name = input("please input your name: ")


count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = 'faces/'+name+str(count)+'.jpg'          
        cv2.imwrite(file_name_path,face)
        
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
        print('얼굴 저장 완료' + str(count))
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==500:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')

data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):    
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue    
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)


if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

model.write('trainer/' + name +'.yml')

DeleteAllFiles('faces/')

print("Model Training Complete!!!!!")


models = {}

model_dir = 'trainer'

yaml_files = glob.glob(os.path.join(model_dir, '*.yml'))
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

x,y,w,h = 0,0,0,0
def face_detector(img, size = 0.5):
    global x,y,w,h
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi  


for yaml_file in yaml_files:
    model_name = os.path.splitext(os.path.basename(yaml_file))[0]
    model = cv2.face_LBPHFaceRecognizer.create()
    model.read(yaml_file)
    models[model_name] = model


cam = cv2.VideoCapture(0)

while 1:
    ret, frame = cam.read()
    
    image, face = face_detector(frame)
    face_confidence = {}
    try:
        for model_name, model in models.items():
            test_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            label, result = model.predict(test_gray)
            if result < 100:
                confidence = int(100*(1-(result)/300))
                face_confidence[model_name] = confidence
            else:
                label_text = 'Unknown'
        molel_name = max(face_confidence, key=face_confidence.get)
        model_confidence = face_confidence[molel_name]
        if model_confidence > 50:
            cv2.putText(image, str(model_confidence)+"%", (x+80, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, molel_name, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    except:
        pass
    
    cv2.imshow('Face Cropper', image)
    if cv2.waitKey(1)==13:
        break
cam.release()
cv2.destroyAllWindows()
