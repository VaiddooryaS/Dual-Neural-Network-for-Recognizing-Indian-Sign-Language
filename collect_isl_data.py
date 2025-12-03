import cv2
import mediapipe as mp
import numpy as np
import os

# --- Configuration ---
DATA_PATH = "ISL_Data"
# Actions (signs) to collect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space'])
# Number of samples (frames) to collect for each action
num_samples_per_action = 100

# --- Setup Folders ---
# This part is moved to be inside the collection loop to create folders on demand

# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)
action_index = 0  # Start with the first action

# Set mediapipe model
with mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:


    while cap.isOpened() and action_index < len(actions):
        action = actions[action_index]
        
        # Display info and wait for user to start
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Process frame to show landmarks continuously
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Display instructions
            cv2.putText(image, f"Ready to collect for: '{action}'", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Press 'c' to start collection.", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                action_index = len(actions) # To exit outer loop
                break
            if key & 0xFF == ord('c'):
                # Ensure the main data folder exists
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)
                break
        
        if action_index >= len(actions):
            break



        # --- 5-second countdown before starting burst capture ---
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Get ready! Recording for '{action}' starts in {i}...", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            cv2.waitKey(1000)
        
        # --- 20-second burst capture ---
        action_samples = []
        frame_count = 0
        while frame_count < num_samples_per_action:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            # Process frame and extract keypoints
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and extract keypoints with handedness
            left_hand_kps = np.zeros(21*3)
            right_hand_kps = np.zeros(21*3)
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    handedness = results.multi_handedness[i].classification[0].label
                    landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
                    wrist_coords = landmarks[0]
                    normalized_landmarks = landmarks - wrist_coords
                    landmarks_flat = normalized_landmarks.flatten()

                    if handedness == 'Left':
                        left_hand_kps = landmarks_flat
                    elif handedness == 'Right':
                        right_hand_kps = landmarks_flat
            
            keypoints = np.concatenate([left_hand_kps, right_hand_kps])
            action_samples.append(keypoints)
            frame_count += 1

            # Display recording status
            cv2.putText(image, f"RECORDING for '{action}'", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Frame {frame_count}/{num_samples_per_action}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            # Pace the capture to be ~20 seconds (100 frames / 5 fps = 20s)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                action_index = len(actions)
                break
        
        if action_index >= len(actions):
            break

        # Save all samples for the action into a single file
        if len(action_samples) > 0:
            npy_path = os.path.join(DATA_PATH, action + '.npy')
            np.save(npy_path, np.array(action_samples))
            print(f"Saved {len(action_samples)} samples to {npy_path}")
        
        action_index += 1

    cap.release()
    cv2.destroyAllWindows()