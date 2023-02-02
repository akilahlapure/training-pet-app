import cv2
import numpy as np
import mediapipe as mp
import imutils

from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

cap = cv2.VideoCapture(0)



########################################### FUNCTIONS ###########################################
def change_dog(action):
    dog_sit = 'assets/dog_sit.png'
    dog_laydown = 'assets/dog_laydown.png'
    dog_up = 'assets/dog_up.png'
    dog_stay = 'assets/dog_stay.png'
    dog_spin = 'assets/dog_spin.png'

    if (action == "sit"):
        return dog_sit
    elif (action == "laydown"):
        return dog_laydown
    elif (action == "up"):
        return dog_up
    elif (action == "stay"):
        return dog_stay
    elif (action == "spin"):
        return dog_spin
    else:
        return dog_sit # default image
        


def open_camera():
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                ) 
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    # Actions that we try to detect
    actions = np.array(['sit', 'laydown', 'spin', 'up', 'stay'])

    # Load training model
    model = load_model('action2.h5')

    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    action = 'assets/dog_sit.png'

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=750, height=300)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                action = change_dog(actions[np.argmax(res)])
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                
            cv2.rectangle(image, (0,0), (750, 40), (45, 91, 184), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Read dog and resize
            dog = cv2.imread(action)
            size = 200
            dog = cv2.resize(dog, (size, size))
            
            # Create a mask of dog
            img2gray = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

            # Region of Image (ROI), where we want to insert image
            roi = image[-size-10:-10, -size-10:-10]
        
            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += dog

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def open_instructions():
    root.destroy()

    instructions_window = Tk()
    instructions_window.title("Instructions")

    instructions_window.configure(bg='#F6EDD9')

    canvas = Canvas(instructions_window, width = 750, height = 500, bg="#F6EDD9")
    canvas.pack(expand=YES)

    img1 = PhotoImage(file = "assets/action_up1.png")
    img2 = PhotoImage(file = "assets/action_stay1.png")
    img3 = PhotoImage(file = "assets/action_spin1.png")
    img4 = PhotoImage(file = "assets/action_sit1.png")
    img5 = PhotoImage(file = "assets/action_laydown1.png")

    Label(text="Here are some training signals you can practice!", font=("Raleway bold", 18, "bold"), bg="#F6EDD9").place(x=100, y=25)

    Label(instructions_window, image=img1, bg="#F6EDD9").place(x=130, y=64)
    Label(text="Palm facing up, bring both arms\n upwards.", font=("Raleway", 8), bg="#F6EDD9").place(x=130, y=216)

    Label(instructions_window, image=img2, bg="#F6EDD9").place(x=300, y=64)
    Label(text="Keep palm facing forward.", font=("Raleway", 8), bg="#F6EDD9").place(x=316, y=216)

    Label(instructions_window, image=img3, bg="#F6EDD9").place(x=470, y=64)
    Label(text="Spin your index finger\n clockwise", font=("Raleway", 8), bg="#F6EDD9").place(x=493, y=216)

    Label(instructions_window, image=img4, bg="#F6EDD9").place(x=215, y=254)
    Label(text="Palm facing up, bring one arm\n upwards", font=("Raleway", 8), bg="#F6EDD9").place(x=215, y=406)

    Label(instructions_window, image=img5, bg="#F6EDD9").place(x=385, y=254)
    Label(text="Palm facing down, bring one\n arm downwards", font=("Raleway", 8), bg="#F6EDD9").place(x=385, y=406)

    # Button
    open_cam_text = tk.StringVar()
    open_cam_btn = tk.Button(instructions_window, textvariable=open_cam_text, command=lambda:open_camera(), font="Raleway", bg="#b85c2d", fg="white", width=15)
    open_cam_text.set("START")
    open_cam_btn.place(x=300, y=447)

    instructions_window.mainloop()

#################################### START WELCOME SCREEN #################################### 
root = tk.Tk()
root.title("Welcome!")

canvas = tk.Canvas(root, width=750, height=500, bg="#F6EDD9")
canvas.grid(columnspan=3, rowspan=3)

# Logo
logo = Image.open('assets/paw_logo.png')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo, bg="#F6EDD9")
logo_label.image = logo
logo_label.place(x=287, y=94)

# Slogan
slogan = tk.Label(root, text="Train yourself to train your pet!", font=("Raleway", 18, "bold"), bg="#F6EDD9")
slogan.place(x=190, y=300)

# Button
open_instruc_text = tk.StringVar()
open_instruc_btn = tk.Button(root, textvariable=open_instruc_text, command=lambda:open_instructions(), font="Raleway", bg="#b85c2d", fg="white", width=15)
open_instruc_text.set("Get Started")
open_instruc_btn.place(x=290, y=350)
# open_instruc_btn.grid(column=1, row=2)

# My name
name = tk.Label(root, text="By: Akilah Antoinnette G. Lapure", font=("Raleway", 8), bg="#F6EDD9")
name.place(x=300, y=462)

# canvas = tk.Canvas(root, width=750, height=100, bg="#F6EDD9")
# canvas.grid(columnspan=3)


root.mainloop()
#################################### END WELCOME SCREEN #################################### 