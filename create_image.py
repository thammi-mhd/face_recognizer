import os
import cv2

name = input("enter the student name: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    
    cv2.imshow("captures Images", frame)
    
    img_path = f"{path}/{count}.jpg"
    cv2.imwrite(img_path, frame)
    
    count += 1
    
    if count == 30:
        break
    
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"images are created for {name}")