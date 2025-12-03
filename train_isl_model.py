
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Configuration (should match collect_isl_data.py) ---
DATA_PATH = "ISL_Data"
# Actions (signs) that were collected
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space']) # Make sure this matches the folders in ISL_Data
# Number of samples collected for each action
num_samples_per_action = 100

# --- Data Loading and Preprocessing ---

# Create a mapping from action name to a number
label_map = {label: num for num, label in enumerate(actions)}

all_landmarks = []
all_labels = []
for action in actions:
    try:
        res = np.load(os.path.join(DATA_PATH, action + ".npy"))
        all_landmarks.append(res)
        # Create a label array of the correct size for this action's data
        all_labels.append(np.full(res.shape[0], label_map[action]))
    except FileNotFoundError:
        print(f"Warning: Data file not found for action '{action}'. Skipping.")

# Concatenate all landmarks and labels
X = np.concatenate(all_landmarks, axis=0)
y = to_categorical(np.concatenate(all_labels, axis=0)).astype(int)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Split the landmark data into two hands for the multi-input model
X_train_hand1 = X_train[:, :63]
X_train_hand2 = X_train[:, 63:]
X_test_hand1 = X_test[:, :63]
X_test_hand2 = X_test[:, 63:]

# --- Build the Multi-Input Model ---

# Define the input layers for each hand (21 landmarks * 3 coords = 63)
input_hand_1 = Input(shape=(63,), name='hand_1_input')
input_hand_2 = Input(shape=(63,), name='hand_2_input')

# --- Branch for the first hand ---
hand_1_branch = Dense(128, activation='relu')(input_hand_1)
hand_1_branch = Dropout(0.5)(hand_1_branch)
hand_1_branch = Dense(64, activation='relu')(hand_1_branch)

# --- Branch for the second hand ---
hand_2_branch = Dense(128, activation='relu')(input_hand_2)
hand_2_branch = Dropout(0.5)(hand_2_branch)
hand_2_branch = Dense(64, activation='relu')(hand_2_branch)

# --- Merge the two branches ---
combined = concatenate([hand_1_branch, hand_2_branch])

# --- Head of the model ---
head = Dense(64, activation='relu')(combined)
head = Dropout(0.5)(head)
output = Dense(actions.shape[0], activation='softmax')(head)

# Create the final model
model = Model(inputs=[input_hand_1, input_hand_2], outputs=output)

model.summary()

# --- Compile and Train the Model ---

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Use a callback to stop training early if performance plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit([X_train_hand1, X_train_hand2], y_train, epochs=200, validation_data=([X_test_hand1, X_test_hand2], y_test), callbacks=[early_stopping])

# --- Save the Model ---

model.save('isl_model.h5')
print("\nModel saved as isl_model.h5")
