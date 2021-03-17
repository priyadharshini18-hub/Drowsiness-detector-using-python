import cv2
import dlib
import numpy as np
from scipy.spatial import distance

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def calculate_MAR(mouth1):
    A = distance.euclidean(mouth1[1], mouth1[7])
    B = distance.euclidean(mouth1[2], mouth1[6])
    C = distance.euclidean(mouth1[3], mouth1[5])
    D = distance.euclidean(mouth1[0], mouth1[4])
    mouth_aspect_ratio = (A+B+C)/(3.0*D)
    return mouth_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("/Users/priya/PycharmProject/drowsiness/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth = []

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []


        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)


        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)


        # for lips...

        for n in range(60, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            mouth.append((x, y))
            next_point = n + 1
            if n == 67:
                next_point = 60
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)



        m=round(calculate_MAR(mouth),2)
        print(m)


        if EAR<0.26 or m>0.5:
            cv2.putText(frame,"DROWSY",(500,80),
                cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
            cv2.putText(frame,"Are you Sleepy?",(20,600),
                cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
            print("Drowsy")
        #print(EAR)

    cv2.imshow("Are you Sleepy....", frame)

    #this will capture video in Black and white...
    #cv2.imshow( 'frame', gray)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()