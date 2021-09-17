import face_recognition
import cv2
import numpy as np

#Loading images and Converting it into RGB

imgAtharva = face_recognition.load_image_file('C:/Users/Abhinav/Downloads/Smart attendance Project/Atharva Normal.jpg')
imgAtharva = cv2.cvtColor(imgAtharva,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('C:/Users/Abhinav/Downloads/Smart attendance Project/Elon-Musk-660x480.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#Face Locations and encodings
faceloc = face_recognition.face_locations(imgAtharva)[0]
encodeAtharva = face_recognition.face_encodings(imgAtharva)[0]
cv2.rectangle(imgAtharva,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeAtharvaTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,255,0),2)
#Viewing Image

results = face_recognition.compare_faces([encodeAtharva],encodeAtharvaTest)
facedis = face_recognition.face_distance([encodeAtharva],encodeAtharvaTest)
cv2.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
cv2.imshow('Atharva', imgAtharva)
cv2.namedWindow('Test Atharva',cv2.WINDOW_NORMAL)
cv2.imshow('Test Atharva', imgTest)
cv2.waitKey(0)