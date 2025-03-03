import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from XCeptionV1 import Xception
from face_preprocessing import FacePreprocessor

# Emotion categories for FER-2013
torch.manual_seed(8)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = Xception(num_classes=len(EMOTIONS))
        self.model = self.model.to(self.device)
        
        #Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.face_preprocessor = FacePreprocessor()

    def predict_emotion(self, face):
        """
        Predict the emotion from a given face image.
        param face: Cropped face image as a NumPy array.
        return: Predicted emotion label and probabilities.
        """
        if face is None:
            return None, None
        
        #Convert NumPy image to PIL and apply transformation
        face_pil = Image.fromarray(face)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            output = self.model(face_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  #Convert to probabilities
            predicted_class = torch.argmax(output, dim=1).item()  #Get class index
        
        return EMOTIONS[predicted_class], probabilities

    def predict_emotions_from_frames(self, frames):
        """
        Predict emotions from x frames of images.
        param frames: List of x frames (NumPy arrays) from the webcam.
        return: List of detected emotions for each frame.
        """
        emotions = []
        
        for frame in frames:
            face, _ = self.face_preprocessor.preprocess_face(frame)
            if face is not None:
                emotion, _ = self.predict_emotion(face)
                emotions.append(emotion)
            else:
                emotions.append("No Face Detected")
        
        return emotions
