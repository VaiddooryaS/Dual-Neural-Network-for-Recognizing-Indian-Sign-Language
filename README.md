
[ISL_recognition_report.pdf](https://github.com/user-attachments/files/23938429/ISL_recognition_report.pdf)



1. Built a dual-input deep neural network that processes 63 landmark coordinates per hand, enabling accurate recognition of both single-hand and two-hand ISL gestures using TensorFlow.

2. Implemented real-time computer vision using MediaPipe to extract 3D hand landmarks and normalize them relative to the wrist, ensuring position- and scale-invariant gesture recognition.

3. Designed a full data pipeline including dataset collection, landmark preprocessing, class labeling, one-hot encoding, and train–test splitting to support efficient multi-class gesture classification.

4. Trained a dense neural architecture with dropout regularization, EarlyStopping, and ModelCheckpoint callbacks, achieving 97.8% validation accuracy across 27 ISL gesture classes.

5. Developed a real-time GUI using Tkinter + OpenCV to capture webcam feeds, display predicted signs, accumulate sentences, and provide auditory feedback through text-to-speech.

6. Integrated translation capabilities using Googletrans, enabling instant conversion of recognized ISL sentences into multiple regional Indian languages within the same application.
