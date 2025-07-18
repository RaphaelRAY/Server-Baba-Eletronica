from ultralytics import YOLO
import cv2

class VideoProcessor:
    def __init__(self, camera_handler):
        self.camera = camera_handler

        # Carrega modelo leve YOLOv8n (pré-treinado para detecção de pessoas)
        self.model = YOLO("yolo11n.pt")  

    def process_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return None

        # Roda inferência (classe 0 = pessoa)
        results = self.model.predict(source=frame, conf=0.4, classes=[0], verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = f"Pessoa {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        return frame
    
    def process_frame_data(self, frame):
        """Recebe um frame e retorna resultados da inferência."""
        if frame is None:
            return None
        results = self.model.predict(source=frame, conf=0.4, classes=[0], verbose=False)
        return results

    def get_processed_frame(self):
        """Return the latest frame from the camera."""
        return self.camera.get_frame()

