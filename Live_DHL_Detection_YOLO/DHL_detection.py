from ultralytics import YOLO
import cv2
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import requests

'''
Function for sending a Notifivation to my phone everytime a detection has been made
'''
def send_pushover_notification(user_key, app_token, message, image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": app_token,
                "user": user_key,
                "message": message,
            },
            files={
                "attachment": image_file
            }
        )
    return response.status_code == 200


'''
Loading the finetuned model, creating log file and accessing API Keys
'''
model = YOLO('path/to/your/dhl_model.pt')
log_file = open("dhl_detections_log.txt", "a")
load_dotenv()
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN")


'''
Live Detection of DHL Cars
'''
#cap = cv2.VideoCapture("/Users/felixfautsch/Downloads/IMG_1908.JPG")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)

            log_entry = f"[{timestamp}] Detected: {cls_name}, BBox: {bbox}"
            print(log_entry)
            log_file.write(log_entry + "\n")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if result.boxes:  
                filename = f"detection_{timestamp.replace(':', '-')}.jpg"
                cv2.imwrite(filename, frame)

                success = send_pushover_notification(
                    user_key=PUSHOVER_USER_KEY,
                    app_token=PUSHOVER_API_TOKEN,
                    message=f"Detected: {cls_name} at {timestamp}",
                    image_path=filename
                )

                if success:
                    print("Pushover notification sent.")
                else:
                    print("Failed to send Pushover notification.")

    cv2.imshow("YOLO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()