import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

aditya_image = face_recognition.load_image_file("Photos/Aditya.jpeg")
aditya_encoding = face_recognition.face_encodings(aditya_image)[0]

aman_image = face_recognition.load_image_file("Photos/Aman.jpeg")
aman_encoding = face_recognition.face_encodings(aman_image)[0]

bhoomi_image = face_recognition.load_image_file("Photos/Bhoomi.jpeg")
bhoomi_encoding = face_recognition.face_encodings(bhoomi_image)[0]

nabhanya_image = face_recognition.load_image_file("Photos/Nabhanya.jpeg")
nabhanya_encoding = face_recognition.face_encodings(nabhanya_image)[0]

mohit_image = face_recognition.load_image_file("Photos/Mohit.jpeg")
mohit_encoding = face_recognition.face_encodings(mohit_image)[0]

known_face_encoding = [
aditya_encoding,
aman_encoding,
bhoomi_encoding,
nabhanya_encoding,
mohit_encoding

]

known_faces_names = [
"Aditya Singh Sikarwar",   
"Aman Phaltankar",   
"Bhoomi Gupta",  
"Nabhanya Singh",   
"Mohit Mendiratta"  
 
]

students = known_faces_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f= open(current_date+'.csv','w+',newline = '')
Inwriter = csv.writer(f)

print("These are all students in B'Tech section(A)")
print(students)
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
       
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding ,face_encoding)
            name=""
        
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)

            if name in known_faces_names:
                if name in students :
                   
                    students.remove(name)
                    print(students)
                    time_now= datetime.now()
                    current_time = time_now.strftime("%H:%M:%S")
                    Inwriter.writerow([name,current_time])
            

    cv2.imshow("Attendence system", frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        if students==known_faces_names:
            print("All Students are Absent Today")
        else:
          print("These Students are absent today--> ", students)
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
