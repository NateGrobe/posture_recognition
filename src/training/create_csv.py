'''
    Builds a csv file with a list of pose landmarks for each image
'''

import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import os

# determines the name of the folder given a file prefix
def get_folder(img_class: str) -> str:
    if img_class == 'average':
        return 'Average'
    elif img_class == 'bendingdown':
        return 'BendingDown'

def create_csv():
    mp_holistic = mp.solutions.holistic

    output_file = './models/coords_test.csv'
    avg_dir = "./Average"
    bd_dir = "./BendingDown"
    results = None
    first_loop = True

    s_count = 0
    f_count = 0

    # make sure there are only jpegs being read in
    avg_imgs = [x for x in os.listdir(avg_dir) if ".jpg" in x]
    bd_imgs = [x for x in os.listdir(bd_dir) if ".jpg" in x]
    img_list = avg_imgs + bd_imgs

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for image_path in img_list:

            img_class = image_path.split('_')[0].lower()
            image = cv2.imread(f"./{get_folder(img_class)}/{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            # Adds the column names to the csv file
            # Done this way because depending on the image set
            # the number of columns is variable
            if first_loop:
                # generate header
                num_coords = len(results.pose_landmarks.landmark)
                landmarks = ['class']
                for i in range(1, num_coords + 1):
                    landmarks += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]

                # write to file
                with open(output_file, mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)

                first_loop = False

            # Extract pose landmarks from images and write to csv
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Append class name 
                row.insert(0, img_class)
                
                # Export to CSV
                with open(output_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 

                s_count += 1
            except:
                f_count += 1
        

    print(f"{s_count} success")
    print(f"{f_count} failed")

if __name__ == '__main__':
    create_csv()
