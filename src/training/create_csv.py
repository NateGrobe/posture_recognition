'''
A simple interface for taking photos of a pose.

This gives the user 3 seconds to get into position. Once in position,
60 photos will be taken and then the application will close.
'''

import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import os

if __name__ == '__main__':
    title = ""
    
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # cap = cv2.VideoCapture(0)
    # get the path/directory
    folder_dir = "./Average"
    for images in os.listdir(folder_dir):
 
        
        results = None
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            


            results = holistic.process(images)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

            mp_drawing.draw_landmarks(images, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

            # export coordinates
            # put this in its own file
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concatenate rows
                row = pose_row+face_row
                
                # Append class name 
                row.insert(0, title)
                
                # Export to CSV
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 
            except:
                pass


