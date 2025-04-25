import cv2
import numpy as np
import torch
from deepface import DeepFace
import dlib

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

detector = dlib.get_frontal_face_detector()

def detect_objects(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    
    detections = results.xyxy[0].numpy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5: 
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image



def detect_facial_expressions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        face_img = image[face.top():face.bottom(), face.left():face.right()]
        if face_img.size > 0:
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                
                emotion = result[0]['dominant_emotion']
                
                cv2.rectangle(image, (face.left(), face.top()), 
                            (face.right(), face.bottom()), (0, 0, 255), 2)
                
                cv2.putText(image, emotion, (face.left(), face.top()-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except:
                pass  # Handle cases where DeepFace fails to analyze the face image
    
    return image


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame = detect_objects(frame)
        
        frame = detect_facial_expressions(frame)
        
        cv2.imshow("Real-time Recognition", frame)
        
        # ro Exit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()