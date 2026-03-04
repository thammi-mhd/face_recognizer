import cv2
import face_recognition
import pickle
from datetime import datetime
import numpy as np

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
    def mark_attendence(name):
        with open("attendence.csv", "r+") as f:
            lines = f.readlines()
            names = []
            
            for line in lines:
                entry = line.split(",")
                names.append(entry[0])
                
            if name not in names:
                now = datetime.now()
                time = now.strftime("%H:%M:%S")
                
                f.writelines(f"\n{name},{time}")
                
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data["names"][best_match_index]
            mark_attendence(name)
            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    cv2.imshow("Attendence System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()