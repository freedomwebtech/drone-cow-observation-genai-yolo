import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Manually set Google API Key
GOOGLE_API_KEY = "AIzaSyDR6iX-Qu_eXvXYUUHG95cSF74msqoQzYg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  

class CowDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="best.pt"):
        """Initializes the cow detection processor."""
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names  # Object class names detected by YOLO
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        self.cy1 = 501
        self.offset = 8
        self.processed_track_ids = set()
        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"cow_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_cows"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Cow Color\n")
                file.write("-" * 50 + "\n")

    def analyze_image_with_gemini(self, image_path):
        """Analyzes a cropped cow image using Gemini AI and extracts only the cow's color."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Explicitly ask for color detection
            message = HumanMessage(
                content=[
                    {"type": "text", "text":  "Identify only the color of the cow in this image and return the result in a table format.\n\n"
                 "| Cow Color |\n"
                 "|-----------|\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Cow image for color detection"}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def process_crop_image(self, image, track_id):
        """Saves cropped cow images and analyzes them for color only."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"cow_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)

        cow_color = self.analyze_image_with_gemini(image_filename)

        # ✅ Print Gemini AI response
        print(f"\n🔍 Cow Color for Track ID {track_id}: {cow_color}\n")

        # ✅ Save cow color to the text file
        with open(self.output_filename, "a", encoding="utf-8") as file:
            file.write(f"{timestamp} | Track ID: {track_id} | {cow_color}\n")

        print(f"✅ Data saved for track ID {track_id}.")

    def crop_and_process(self, frame, box, track_id):
        """Crops cow image from frame and processes it."""
        if track_id in self.processed_track_ids:
            print(f"Track ID {track_id} already processed. Skipping.")
            return  

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)

        # Start a separate thread for processing the image
        threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        """Processes each video frame for cow detection."""
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                detected_class = self.names[class_id]  

                # Ensure only cows are processed
                if detected_class.lower() != "cow":
                    continue  

                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) // 2
                
                if self.cy1 < (cy + self.offset) and self.cy1 > (cy - self.offset):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, detected_class, (x1, y1), 1, 1)

                    # ✅ Send only cow images for color detection
                    self.crop_and_process(frame, box, track_id)

        return frame
    
    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        """Tracks mouse movement over the OpenCV window."""
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        """Starts the cow detection and tracking process."""
        cv2.namedWindow("Cow Detection")
        cv2.setMouseCallback("Cow Detection", self.mouse_callback)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_video_frame(frame)
            cv2.line(frame, (0, 501), (1019, 501), (0, 0, 255), 2)
            cv2.imshow("Cow Detection", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):  
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Data saved to {self.output_filename}")

if __name__ == "__main__":
    video_file = "cow1.mp4"
    processor = CowDetectionProcessor(video_file)
    processor.start_processing()
