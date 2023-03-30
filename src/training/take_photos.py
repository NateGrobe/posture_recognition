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
    title = "Insert title here"
    
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)

    results = None

    st = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        start_time = time.time()

        counter = 0

        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # # 1. Draw face landmarks
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
            #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            #                         )
            
            # # 2. Right hand
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            #                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            #                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            #                         )

            # # 3. Left Hand
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            #                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            #                         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            #                         )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
                            
            cv2.imshow(title, image)

            if time.time() - st > 3:
                counter += 1
                try:
                    os.mkdir(title)
                except:
                    pass

                path = f"./{title}/{title}_{counter}.jpg"
                cv2.imwrite(path, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # export coordinates
            # put this in its own file
            # try:
            #     # Extract Pose landmarks
            #     pose = results.pose_landmarks.landmark
            #     pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
            #     # Extract Face landmarks
            #     face = results.face_landmarks.landmark
            #     face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
            #     # Concatenate rows
            #     row = pose_row+face_row
                
            #     # Append class name 
            #     row.insert(0, title)
                
            #     # Export to CSV
            #     with open('coords.csv', mode='a', newline='') as f:
            #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #         csv_writer.writerow(row) 
            # except:
            #     pass

    cap.release()
    # cap.destroyAllWindows()


    num_coords = len(results.pose_landmarks.landmark) 
    # + len(results.face_landmarks.landmark)

