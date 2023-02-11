import pickle
import cv2
import numpy as np

def load_data():
    file = open('database/pickled', 'rb')
    data = pickle.load(file)
    file.close()
    return data


def remove_person(data):
    
    pixel_list = data[0]
    indx_list = data[1]
    name_list = data[2]
    id_list = data[3]
    
    print(name_list)
    
    while True:
        print("\nWho do you want to remove?\n")
        rm_name = input()
        
        if rm_name in name_list:
            print('is in')
            
            indx = name_list.index(rm_name) #index of removing name in name list
            
            new_indx_list = []
            new_pixel_list = []
            new_name_list = []
            new_id_list = []
            
            for i in range(len(indx_list)):
                if indx_list[i] != indx:
                    new_pixel_list.append(pixel_list[i])
                    new_indx_list.append(indx_list[i])
        
            
            for i in range(len(new_indx_list)):
                if new_indx_list[i] > indx:
                    new_indx_list[i] -= 1
            
            name_list.remove(rm_name)
            id_list.remove(id_list[indx])
            new_name_list = name_list
            new_id_list = id_list
            
            new_data = [new_pixel_list, new_indx_list, new_name_list, new_id_list]
            return new_data
        else:
            print('not here')
    
def model_train(faces, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    label = np.array(labels)
    face_recognizer.train(faces, label)
    face_recognizer.write(r"created_models/trainer.yml")
    print("\n [INFO] {0} faces trained.".format(len(np.unique(label))))
    
data = load_data()
data = remove_person(data)
model_train(data[0], data[1])


    
    
