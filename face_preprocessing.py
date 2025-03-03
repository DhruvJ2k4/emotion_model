import cv2
import mediapipe as mp
import numpy as np

class FacePreprocessor:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def preprocess_face(self, frame):
        """ 
        Detects and extracts the face from the given frame. 
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                # x, y, width, height = (
                #     int(bboxC.xmin * w),
                #     int(bboxC.ymin * h),
                #     int(bboxC.width * w),
                #     int(bboxC.height * h),
                # )
                
                #Expanding bounding box by 10%
                x, y, width, height = (
                    int(bboxC.xmin * w),
                    int(bboxC.ymin * h),
                    max(int(bboxC.width * w), 48),
                    max(int(bboxC.height * h), 48)
                )

                face = frame[y:y + height, x:x + width]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                return face_gray, (x, y, width, height)
        
        return None, None
