import cv2
import os
import face_recognition
import pickle

dataset_path = "dataset"

known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

data = {"encodings": known_encodings, "names": known_names}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Face encodings created successfully")