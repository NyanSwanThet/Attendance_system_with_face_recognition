import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

    
def load_data():
    file = open('database/pickled', 'rb')
    data = pickle.load(file)
    file.close()
    return data



def attendance(df, nam, uid):
    
    
    nam_list = df['Name'].to_list()
    
    is_new_name = True
    for i in nam_list:
        if nam == i:
            is_new_name = False
            
    time = datetime.now()
    current_time = time.strftime("%H:%M:%S")
            
    if is_new_name == True:  #if name is not in excel, add name 
        
        new = pd.DataFrame({'Name':[nam], 'UID':[uid], 'Time':[current_time]})
        
        df = pd.concat([df, new], ignore_index = True)
        
        df.to_excel('attendance.xlsx')
        return df
    
    else:           #if name is in excel, update time    
        same_nam = (df['Name'] == nam) .to_list()
        df['Time'].loc[same_nam] = current_time
        return df



face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"created_models/trainer.yml")

data = load_data()

df = pd.DataFrame({'Name':[], 'UID':[], 'Time':[]})

#index 0->pixels  1->labels  2->names  3->UID
names = data[2]
uids = data[3]

font = cv2.FONT_HERSHEY_SIMPLEX

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    
    modelFile = r"models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = r"models/deploy.prototxt.txt"
    
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = frame.shape[:2]
    
    resized = cv2.resize(frame, (300, 300))
    
    #change image to binary large object (blob)
    #parameters are image, scalefactor, output_size, (mean_r, mean_g, mean_b)
    blob = cv2.dnn.blobFromImage(resized, 1.0, (300, 300), (104.0, 117.0, 123.0)) 
    net.setInput(blob)
    faces = net.forward()
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                
                face = frame[ y:y1, x:x1, :]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = np.array(face, 'uint8')
                label, confidence = face_recognizer.predict(face)
                
                # If confidence is less them 100 ==> "0" : perfect match 
                if (confidence < 60):
                    
                    nam = names[label]
                    uid = uids[label]
                    
                    
                    df = attendance(df, nam, uid)
                    
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                else:
                    nam = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                
                cv2.putText(
                            frame,                
                            str(nam)+' '+str(confidence),  #text
                            (x+5,y-5),              #coordinate x and y
                            font,                   #font
                            1,                      #font scale
                            (255,255,255),          #color
                            2                       #thickness
                           )
                
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()