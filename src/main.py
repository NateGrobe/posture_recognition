from tkinter import *
import customtkinter
import pickle
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.start = False
        # Window Configuration
        self.title("Posture Recognition")
        self.geometry(f"{1100}x{725}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets  OPTIONS COL
        self.sidebar_frame = customtkinter.CTkFrame(
            self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Options", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # START BUTTON
        self.startbtn = customtkinter.CTkButton(
            self.sidebar_frame, text="Start Monitor", command=self.handle_start_stop)
        self.startbtn.grid(row=1, column=0, padx=20, pady=10)

        # LIGHT AND DARK MODE ETC
        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=6, column=0, padx=20, pady=(10, 10))

        # OUTPUT VIDEO

        self.cameraFrame = customtkinter.CTkFrame(
            self, width=950, corner_radius=0)
        self.cameraFrame.grid(row=0, column=1, padx=10, pady=10)
        self.cam = customtkinter.CTkLabel(self.cameraFrame, text="")
        self.cam.grid()

        # media pipe setup
        self.mp_drawing = mp.solutions.drawing_utils  # drawing helpers
        self.mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

        self.blf = open("models/body_language.pkl", "rb")
        self.pose_model = None
        with open("models/body_language.pkl", "rb") as f:
            self.pose_model = pickle.load(f)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def handle_start_stop(self):
        if not self.start:
            self.start = True
            self.startbtn.configure(text="Stop")
            self.on_start()
            self.start_main()
        else:
            self.start = False
            self.startbtn.configure(text="Start")
            self.stop_main()
            self.on_stop()

    def start_main(self):
        self.start = True
        self.main()

    def stop_main(self):
        self.start = False

    def on_start(self):
        self.cap = cv2.VideoCapture(0)

    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()

    def main(self):
        if self.start:
            results = None

            ret, img = self.cap.read()
            if ret:
                results = self.holistic.process(img)

                # 4. Pose Detections
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(
                                                   color=(245, 117, 66), thickness=2, circle_radius=4),
                                               self.mp_drawing.DrawingSpec(
                                                   color=(245, 66, 230), thickness=2, circle_radius=2),
                                               )

                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = self.pose_model.predict(X)[0]
                    body_language_prob = self.pose_model.predict_proba(X)[0]
                    print(body_language_class, body_language_prob)

                    # Get status box
                    cv2.rectangle(img, (0, 0), (250, 60),
                                  (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(
                        img, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, body_language_class.split(' ')[
                                0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(
                        img, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
                        10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                except:
                    pass

                # cv2.imshow('Raw Webcam Feed', img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image1 = Image.fromarray(img)
                ImgTks = ImageTk.PhotoImage(image=image1)
                # self.canvas.create_image(0, 0, image=ImgTks, anchor=NW)
                self.cam.imgtk = ImgTks
                self.cam.configure(image=ImgTks)

                self.after(1, self.main)


if __name__ == '__main__':
    app = App()
    app.mainloop()
