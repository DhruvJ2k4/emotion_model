import cv2
from collections import Counter
import time
from face_preprocessing import FacePreprocessor
from emotion_model import EmotionModel, EMOTIONS
from collections import Counter

MODEL_PATH = "cnn_full_model.pth"
USER_NAME = "User"
FPS = 30

# Initialize modules
preprocessor = FacePreprocessor()
emotion_model = EmotionModel(MODEL_PATH)

def filter_and_select_emotion(emotions):
    """
    Processes a list of detected emotions:
    Ignores "Neutral" if coupled with any negative emotion.
    Returns the most frequent emotion, giving priority to later occurrences in case of a tie.
    param emotions: List of detected emotions.
    return: The selected dominant emotion.
    """
    NEGATIVE_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}
    
    #Ignore "Neutral" if a negative emotion is present
    filtered_emotions = [e for e in emotions if e != "Neutral" or not any(ne in emotions for ne in NEGATIVE_EMOTIONS)]
    if not filtered_emotions:  #If all were "Neutral", keep it
        filtered_emotions = emotions

    #Count occurrences, giving priority to later occurrences in case of a tie
    emotion_counts = Counter(filtered_emotions)
    most_frequent = max(filtered_emotions, key=lambda e: (emotion_counts[e], emotions[::-1].index(e)))
    return most_frequent

def capture_and_analyze():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Emotion Detection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'c' to capture 10 frames", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            captured_frames = []
            emotions = []
            
            # Capture 10 frames over 5 second
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                time.sleep(0.5)
                
                processed_face, _ = preprocessor.preprocess_face(frame)
                
                if processed_face is not None and processed_face.shape[0] >= 48 and processed_face.shape[1] >= 48:
                    emotion, probabilities = emotion_model.predict_emotion(processed_face)
                    
                    # Fix Sad vs. Angry confusion
                    if emotion == "Sad" and probabilities[EMOTIONS.index("Sad")] < 0.6:
                        if probabilities[EMOTIONS.index("Angry")] > 0.4:
                            emotion = "Angry"  # Adjust misclassification
                    
                    # Ignore uncertain predictions
                    if max(probabilities) < 0.3:
                        print("Uncertain prediction, recapturing...")
                        continue
                    
                    emotions.append(emotion)
                else:
                    print("No face detected or face too small, retrying...")
                    emotions.append("No Face Detected")

            valid_emotions = [e for e in emotions if e != "No Face Detected"]
            if valid_emotions:
                final_emotion = Counter(valid_emotions).most_common(1)[0][0]
            else:
                print("No face detected in all frames, retrying...")
                return capture_and_analyze()


            print("Captured Emotions:", emotions)
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotions

if __name__ == "__main__":
    emotions = capture_and_analyze()
    final_detected_emotion = filter_and_select_emotion(emotions)
    print("Final Detected Emotion:", final_detected_emotion)
