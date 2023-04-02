import tkinter
from tkinter import *
import customtkinter
import pickle
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk, ImageFont, ImageDraw
customtkinter.set_appearance_mode("Dark")
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

        self.radiobutton_frame = customtkinter.CTkFrame(
            self.sidebar_frame, width=140, corner_radius=0)
        self.radiobutton_frame.grid(row=2, column=0, rowspan=4, sticky="nsew")
        self.radio_var = tkinter.IntVar()
        self.radiobutton_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, text="Ridge",
                                                          command=self.change_model, variable=self.radio_var, value=1)
        self.radiobutton_1.grid(
            row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.radiobutton_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, text="Random Forest",
                                                          command=self.change_model, variable=self.radio_var, value=2)
        self.radiobutton_2.grid(
            row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.radiobutton_2.select()

        self.radiobutton_3 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, text="Logistic",
                                                          command=self.change_model, variable=self.radio_var, value=3)
        self.radiobutton_3.grid(
            row=2, column=0, padx=20, pady=10, sticky="nsew")

        self.radiobutton_4 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, text="Boosted",
                                                          command=self.change_model, variable=self.radio_var, value=4)
        self.radiobutton_4.grid(
            row=3, column=0, padx=20, pady=10, sticky="nsew")

        # START BUTTON
        self.startbtn = customtkinter.CTkButton(
            self.sidebar_frame, text="Start Monitor", command=self.handle_start_stop)
        self.startbtn.grid(row=1, column=0, padx=20, pady=10)

        # LIGHT AND DARK MODE ETC
        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=10, column=0, padx=20, pady=(10, 10))

        # OUTPUT VIDEO

        self.cameraFrame = customtkinter.CTkFrame(
            self, width=1920, corner_radius=0)
        self.cameraFrame.grid(row=0, column=1, padx=10, pady=10)
        self.cam = customtkinter.CTkLabel(self.cameraFrame, text="")
        self.cam.grid()

        # media pipe setup
        self.mp_drawing = mp.solutions.drawing_utils  # drawing helpers
        self.mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.blf = open("training/models/rf.pkl", "rb")
        self.rc_model = open("training/models/rc.pkl", "rb")
        self.rf_model = open("training/models/rf.pkl", "rb")
        self.lr_model = open("training/models/lr.pkl", "rb")
        self.gb_model = open("training/models/gb.pkl", "rb")

        self.pose_model = None
        # default model
        with open("training/models/rf.pkl", "rb") as f:
            self.pose_model = pickle.load(f)

        # open all models
        with open("training/models/rc.pkl", "rb") as f:
            self.rc_model = pickle.load(f)
        with open("training/models/rf.pkl", "rb") as f:
            self.rf_model = pickle.load(f)
        with open("training/models/lr.pkl", "rb") as f:
            self.lr_model = pickle.load(f)
        with open("training/models/gb.pkl", "rb") as f:
            self.gb_model = pickle.load(f)

    # change appearance
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    # start stop button event handler
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

    # starts main function
    def start_main(self):
        self.start = True
        self.main()

    # stops main function
    def stop_main(self):
        self.start = False

    # starts camera
    def on_start(self):
        self.cap = cv2.VideoCapture(0)

    # closes camera
    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()
        black_img = np.zeros((720, 960, 3), dtype=np.uint8)
        blk_ImgTks = ImageTk.PhotoImage(image=Image.fromarray(black_img))
        self.cam.configure(image=blk_ImgTks)

    # changes model based on radio buttons
    def change_model(self):
        if self.radio_var == 1:
            self.pose_model = self.rc_model
        elif self.radio_var == 2:
            self.pose_model = self.rf_model
        elif self.radio_var == 3:
            self.pose_model = self.lr_model
        elif self.radio_var == 4:
            self.pose_model = self.gb_model

    # main loop
    def main(self):
        if self.start:
            results = None

            # read in image
            ret, img = self.cap.read()
            if ret:
                results = self.holistic.process(img)

                # Pose Detections
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(
                                                   color=(245, 117, 66), thickness=2, circle_radius=4),
                                               self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Initialize image variables to write text
                font = ImageFont.truetype("Arial Unicode.ttf", 36)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img1 = Image.fromarray(img)
                edit_image = ImageDraw.Draw(im=img1)

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = self.pose_model.predict(X)[0]
                    body_language_prob = self.pose_model.predict_proba(X)[0]
                    print(body_language_class, body_language_prob)

                    # Display Class
                    edit_image.text((155, 12), 'CLASS', ("red"), font=font)
                    edit_image.text((150, 40), body_language_class.split(' ')[
                                    0], ("red"), font=font)

                    # Display Probability
                    edit_image.text((15, 12), 'PROB', ("red"), font=font)
                    edit_image.text((10, 40), str(round(body_language_prob[np.argmax(
                        body_language_prob)], 2)), ("red"), font=font)

                except:
                    pass
                # Remove temp variable
                del edit_image

                # Format into tkinter image and add to GUI
                ImgTks = ImageTk.PhotoImage(image=img1)
                self.cam.imgtk = ImgTks
                self.cam.configure(image=self.cam.imgtk)
                self.after(1, self.main)


if __name__ == '__main__':
    app = App()
    app.mainloop()
