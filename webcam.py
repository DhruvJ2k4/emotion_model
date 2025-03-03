import cv2
import numpy as np
import collections
import time
from face_preprocessing import FacePreprocessor
from emotion_model import EmotionModel

MODEL_PATH = "cnn_full_model.pth"  # Update with the actual model file
USER_NAME = "User"
FPS = 30  # Approximate webcam frame rate
TIME_WINDOW = 5  # Consider last 5 seconds of emotions

# Emotion classification
EMOTION_CATEGORIES = {
    "Positive": ["Happy", "Surprise"],
    "Negative": ["Angry", "Disgust", "Fear", "Sad"],
    "Neutral": ["Neutral"]
}

# Store only last 5 seconds of data
max_frames = FPS * TIME_WINDOW
emotion_history = collections.deque(maxlen=max_frames)

def classify_emotion(emotion):
    """ Classify emotion as Positive, Negative, or Neutral """
    for category, emotions in EMOTION_CATEGORIES.items():
        if emotion in emotions:
            return category
    return "Neutral"


def draw_probabilities(frame, emotions, probabilities):
    """ Displays emotion probabilities as text on the left side of the screen. """
    start_x, start_y = 50, 100
    spacing = 30
    cv2.putText(frame, f"Name: {USER_NAME}", (start_x, start_y - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
        text = f"{emotion}: {prob:.2f}"
        cv2.putText(frame, text, (start_x, start_y + i * spacing), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    preprocessor = FacePreprocessor()
    emotion_model = EmotionModel(MODEL_PATH)
    cv2.namedWindow("Emotion Detection")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_face, bbox = preprocessor.preprocess_face(frame)
        if processed_face is not None:
            emotion, probabilities = emotion_model.predict_emotion(processed_face)
            emotion_category = classify_emotion(emotion)
            emotion_history.append(emotion_category)

            # Draw bounding box and emotion text
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            draw_probabilities(frame, list(EMOTION_CATEGORIES.keys()), probabilities)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
