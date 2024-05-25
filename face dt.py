import cv2
import face_recognition
import os
import csv
from datetime import datetime

known_faces = []
known_names = []

# Load known faces and names from a directory
images_path = r"C:\Users\Dhruv Patel\OneDrive\Desktop\face recognition\photos"
for image_name in os.listdir(images_path):
    image = face_recognition.load_image_file(os.path.join(images_path, image_name))
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(image_name)[0])

# # Create a CSV file for attendance records
# csv_file = open(r"C:\Users\Dhruv Patel\OneDrive\Desktop\face recognition\hariansh.csv", mode='a', newline='')
# csv_writer = csv.writer(csv_file)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            # Insert new face and name
            new_face_name = input("Enter the name for the new face: ")
            face_image = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            known_faces.append(face_image)
            known_names.append(new_face_name)
            
            # Save the new face image
            cv2.imwrite(os.path.join(images_path, new_face_name + ".jpg"), frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Record attendance in the CSV file with date and time
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%Y-%m-%d")
            #csv_writer.writerow([name, current_date, current_time])

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the CSV file and release resources
#csv_file.close()
video_capture.release()
cv2.destroyAllWindows()