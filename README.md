# emotion_model


XCeptionV1.py: provides the Custom Architecture of the XCeption Model used to train on FER2013 dataset for emotion detection, used to be load the model 
                structure for emotion_model.py
                
face_preprocessing.py: preprocesses the face and returns to emotion_model.py file

emotion_model.py: loads the numpy array faces and runs through the model and return emotion(s)

webcam.py: detects emotions of a live feed

webcam1.py: detects emotions from the given number of frames


tech: torch, torchvision, mediapipe, pillow, numpy, opencv-python
