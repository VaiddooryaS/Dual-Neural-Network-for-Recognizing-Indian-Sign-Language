
import tkinter as tk
from tkinter import font, ttk # Import ttk for Combobox
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from googletrans import Translator # Import googletrans Translator

# --- Configuration ---
# Actions (signs) that were trained
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space']) # Make sure this matches your training

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model('isl_model_2.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'isl_model.h5' is in the same directory and was saved from the new training script.")
    exit()

# --- MediaPipe and OpenCV Setup ---S
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation # Background removal disabled

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # Background removal disabled

cap = cv2.VideoCapture(0)

# --- Prediction Logic Variables ---
predictions = [] # This list holds the last few raw predictions for stability analysis
sentence = [] # This list holds the sequence of STABLE recognized letters/signs
threshold = 0.95  # Using a higher threshold for more confidence
last_stable_prediction = "" # Track the last stable prediction to avoid repeats

# --- Google Translate Setup ---
INDIAN_LANGUAGES = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia (Oriya)": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Assamese": "as",
    "Awadhi": "awa",
    "Bhojpuri": "bho",
    "Kashmiri": "ks",
    "Konkani": "gom",
    "Maithili": "mai",
    "Sanskrit": "sa",
    "Sindhi": "sd"
}

# --- GUI Setup ---
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ISL Recognition GUI")
        self.geometry("1600x900")

        self.full_sentence = ""
        self.translator = Translator() # Initialize googletrans Translator

        # --- Configure the main window's grid ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1) # Webcam column
        self.grid_columnconfigure(1, weight=1) # Text column

        # --- Create frames for each column ---
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20)

        # --- Left Frame Widgets (Webcam) ---
        self.webcam_label = tk.Label(self.left_frame)
        self.webcam_label.pack(pady=10)

        # --- Right Frame Widgets (Text and Button) ---
        # --- Right Frame Widgets (Text and Button) ---
        # Predicted Letter
        self.prediction_frame = tk.Frame(self.right_frame)
        self.prediction_frame.pack(pady=20)
        self.prediction_heading_label = tk.Label(self.prediction_frame, text="PREDICTED LETTER: ", font=("Helvetica", 24, "bold"), fg="black")
        self.prediction_heading_label.pack(side=tk.LEFT)
        self.actual_prediction_label = tk.Label(self.prediction_frame, text="...", font=("Helvetica", 24, "bold"), fg="green")
        self.actual_prediction_label.pack(side=tk.LEFT)
        
        # Sentence
        self.sentence_frame = tk.Frame(self.right_frame)
        self.sentence_frame.pack(pady=20)
        self.sentence_heading_label = tk.Label(self.sentence_frame, text="SENTENCE: ", font=("Helvetica", 20), fg="black")
        self.sentence_heading_label.pack(side=tk.LEFT)
        self.actual_sentence_label = tk.Label(self.sentence_frame, text="", font=("Helvetica", 20), wraplength=600, justify=tk.LEFT, fg="blue")
        self.actual_sentence_label.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.right_frame, text="Clear All", command=self.clear_text, font=("Helvetica", 14), bg="red")
        self.clear_button.pack(pady=20)

        self.backspace_button = tk.Button(self.right_frame, text="Backspace", command=self.backspace_action, font=("Helvetica", 14), bg="yellow")
        self.backspace_button.pack(pady=10)

        # --- Translation Widgets ---
        self.language_label = tk.Label(self.right_frame, text="Translate to:", font=("Helvetica", 12))
        self.language_label.pack(pady=(20, 5))

        self.language_combobox = ttk.Combobox(self.right_frame, values=list(INDIAN_LANGUAGES.keys()), state="readonly")
        self.language_combobox.set("Hindi") # Default language
        self.language_combobox.pack(pady=5)

        self.translate_button = tk.Button(self.right_frame, text="Translate", command=self.translate_sentence, font=("Helvetica", 14))
        self.translate_button.pack(pady=10)

        self.translated_label = tk.Label(self.right_frame, text="Translated: ", font=("Helvetica", 16), wraplength=600, justify=tk.LEFT)
        self.translated_label.pack(pady=20)

        self.update_frame()

    def clear_text(self):
        self.full_sentence = ""
        self.translated_label.config(text="Translated: ") # Clear translated text
        global sentence, predictions, last_stable_prediction
        sentence = []
        predictions = []
        last_stable_prediction = ""

    def backspace_action(self):
        if self.full_sentence:
            self.full_sentence = self.full_sentence[:-1] # Remove the last character
            self.translated_label.config(text="Translated: ") # Clear translated text on backspace

    def translate_sentence(self):
        selected_language_name = self.language_combobox.get()
        target_language_code = INDIAN_LANGUAGES.get(selected_language_name)

        if not target_language_code:
            self.translated_label.config(text="Translated: Please select a valid language.")
            return
        if not self.full_sentence.strip():
            self.translated_label.config(text="Translated: Nothing to translate.")
            return

        try:
            # googletrans API call
            translated = self.translator.translate(self.full_sentence, dest=target_language_code)
            self.translated_label.config(text=f"Translated ({selected_language_name}): {translated.text}")
        except Exception as e:
            self.translated_label.config(text=f"Translated: Error - {e}")

    def update_frame(self):
        global predictions, last_stable_prediction, sentence

        ret, frame = cap.read()
        if not ret:
            self.after(15, self.update_frame)
            return

        # Flip for selfie view
        frame = cv2.flip(frame, 1)

        # Make detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        hand_results = hands.process(image)
        # segmentation_results = selfie_segmentation.process(image) # Background removal disabled
        image.flags.writeable = True
        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Use original image as output
        # Background removal (disabled)
        # condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
        # bg_image = np.zeros(image.shape, dtype=np.uint8)
        # output_image = np.where(condition, image, bg_image)

        # Draw ROI box (cosmetic)
        cv2.rectangle(output_image, (50, 50), (600, 450), (0, 255, 0), 2)

        # Prediction logic
        global last_stable_prediction
        display_action = "..."
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints with handedness and normalization
            left_hand_kps = np.zeros(21*3)
            right_hand_kps = np.zeros(21*3)
            if hand_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    handedness = hand_results.multi_handedness[i].classification[0].label
                    landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
                    wrist_coords = landmarks[0]
                    normalized_landmarks = landmarks - wrist_coords
                    landmarks_flat = normalized_landmarks.flatten()
                    if handedness == 'Left':
                        left_hand_kps = landmarks_flat
                    elif handedness == 'Right':
                        right_hand_kps = landmarks_flat

            keypoints = np.concatenate([left_hand_kps, right_hand_kps])
            hand1_landmarks = np.expand_dims(keypoints[:63], axis=0)
            hand2_landmarks = np.expand_dims(keypoints[63:], axis=0)

            # Make prediction
            res = model.predict([hand1_landmarks, hand2_landmarks])[0]
            current_prediction_index = np.argmax(res)
            current_confidence = res[current_prediction_index]
            predicted_action = actions[current_prediction_index]
            
            # Append raw prediction for stability analysis
            predictions.append(current_prediction_index)

            # --- Stability and Sentence Logic ---
            if current_confidence > threshold:
                if predictions[-15:].count(current_prediction_index) >= 12: # More stability: 12 of last 15 frames
                    current_stable_action = actions[current_prediction_index]
                    display_action = current_stable_action
                    
                    if current_stable_action != last_stable_prediction: # Only process if action has changed
                        if current_stable_action == 'space':
                            self.full_sentence += " " # Add a space for new word
                        else: # It's a letter
                            self.full_sentence += current_stable_action # Append letter directly

                    last_stable_prediction = current_stable_action # Always update last_stable_prediction
            else: # If confidence drops or no hands detected
                predictions = predictions[-10:] # Keep some recent predictions
                last_stable_prediction = "" # Reset last prediction to allow re-detection
        else:
            # If no hands are detected, reset the recent predictions
            predictions = predictions[-10:]
            last_stable_prediction = ""


        # Update GUI labels
        self.actual_prediction_label.config(text=display_action)
        self.actual_sentence_label.config(text=self.full_sentence)

        # Convert image for Tkinter
        img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)

        self.after(15, self.update_frame)

    def on_closing(self):
        cap.release()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
