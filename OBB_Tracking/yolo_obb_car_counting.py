import cv2
from ultralytics import solutions

model_path = "yolo11n-obb.pt"
video_path = "path/to/your/video"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

region_points = [(0, 230), (w, 230), (w, 250), (0, 250)] 

counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model=model_path,
    classes=[9, 10], # large vehicle, small vehicle
    tracker="botsort.yaml",
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    im0 = counter.count(im0)  
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()