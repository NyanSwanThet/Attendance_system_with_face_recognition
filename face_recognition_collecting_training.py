import cv2
import time
import numpy as np
import pickle
import pandas as pd

def model_train(faces, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    label = np.array(labels)
    face_recognizer.train(faces, label)
    face_recognizer.write(r"created_models/trainer.yml")
    print("\n [INFO] {0} faces trained.".format(len(np.unique(label))))

def collect_data(name_set, id_set):
    print('\nEnter name\n')
    name = input()
    name_set.append(name)
    print()
    print(name_set)
    print()
    print('\nEnter student id')
    ids = input()
    id_set.append(ids)
    print()
    print(id_set)
    print()
    
    return name_set, id_set

def store_data(face_set, label_set, name_set, id_set):
    
    sets = [face_set, label_set, name_set, id_set]
    
    data = {'Name':name_set, 'UID': id_set,}
    df = pd.DataFrame(data)
    df.index+=1
    df.to_excel("database/attendance_data.xlsx") 
    
    file = open('database/pickled', 'wb')
    pickle.dump(sets, file)
    file.close()

def data_merge(face_set, label_set, name_set, id_set):
    
    file = open('database/pickled', 'rb')
    old_data = pickle.load(file)
    file.close()
    
    new_face_set = old_data[0] + face_set
    
    new_label = []
    
    for i in range(len(label_set)):
        new_label.append( label_set[i] + max(old_data[1]) + 1)
    new_label_set = old_data[1] + new_label
        
    new_name_set = old_data[2] + name_set
    
    new_id_set = old_data[3] + id_set
    
    new_data = [new_face_set, new_label_set, new_name_set, new_id_set]
    
    file = open('database/pickled', 'wb')
    pickle.dump(new_data, file)
    file.close()
    
    return new_data

def store_and_train(face_set, label_set, name_set, id_set):
    
    while True:
        choice = input('Do you want to merge new data with previous data? [y/n]\n')
        
        if choice != 'y' and choice != 'n':
            print('ONlY y or n!!')
        else:
            break
        
    if choice == 'y':
        face_set, label_set, name_set, id_set = data_merge(face_set, label_set, name_set, id_set)
    
    store_data(face_set, label_set, name_set, id_set)
    
    
    model_train(face_set, label_set)
        

def main():
    vid = cv2.VideoCapture(0)
    
    swch = 0
    duration = 0
    start_time = 0
    
    print("Press 's' to start record faces to train model")
    while True:
        ret, frame = vid.read()
        
        modelFile = r"models/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = r'models/deploy.prototxt.txt'
        
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
                    f = frame[ y:y1, x:x1, :]
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    f = np.array(f, 'uint8')
                    
        cv2.imshow('frame', frame)
        
        # if no need to train 
        if swch == 0:
            n_person = 0
            face_set = []
            label_set = []
            name_set = []
            id_set = []
                
        # if need to train, collect data    
        if swch == 1:
            
            face_set.append(f)
            label_set.append(n_person)
            
            now = time.time()
            duration = now - start_time
            
            if duration >= 10:
                swch = 0
                cv2.destroyAllWindows()
                
                name_set, id_set = collect_data(name_set, id_set)
                
                while True:    
                    
                    ans = input('Is there next person? [y/n]\nIf [y], look web cam for 10 second\n')
                    if ans == 'y' or ans == 'n':
                        break
                    else:
                        print('Only y and n please!')
                    
                if ans == 'y':
                    
                    swch = 1
                    n_person += 1
                    duration = 0
                    start_time = time.time()
                
                if ans == 'n':
                    
                    #model_train(face_set, label_set)
                    #store_data(face_set, label_set, name_set, id_set)
                    store_and_train(face_set, label_set, name_set, id_set)
                   
                    break
                    
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print('Starting to record face to train model\nLook web cam for 10 second\n')
            swch = 1
            start_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break
    vid.release()
    cv2.destroyAllWindows()

main()







