import face_recognition
import cv2
#import pyttsx3
import sys 
import os 
#engine = pyttsx3.init()
# Obtain a known face list
# known_face_encodings = [] # english name, chinese name, encoding
path= "/home/deeplearning/project/face_recognition/known_face/" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
files_number = len (files)
known_face_encodings = [ [0] * 3 for i in range(files_number)] # 建立空二维数组
j = 0
for file in files:
    fullname = file.split (".",1) # Cherry_潘老师
    name_en = (fullname [0]).split ("_",1) [0] # Cherry
    name_cn = (fullname [0]).split ("_",1) [1] # 潘老师
    face_image = face_recognition.load_image_file(path+"/"+file)
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings [j][0] = name_en
    known_face_encodings [j][1] = name_cn
    known_face_encodings [j][2] = face_encoding
    j += 1
video_capture = cv2.VideoCapture(0)
process_this_frame = True
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            name_en = "Unknown"
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces([item [2] for item in known_face_encodings], face_encoding,tolerance=0.45)
             # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                face_found = True
                first_match_index = matches.index(True)
                name_en = known_face_encodings[first_match_index][0]
                face_names.append(name_en)
    process_this_frame = not process_this_frame
        # Display the results
    
    for (top, right, bottom, left), name in zip(face_locations,face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # if name != "Unknown" :
        #    name_cn_id = [item [0] for item in known_face_encodings].index(name)
        #    name_cn =  known_face_encodings[name_cn_id][1]
        #engine.say('欢迎'+name_cn) 
        #engine.runAndWait()
        # Display the resulting image
    cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

