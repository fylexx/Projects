import cv2
import numpy as np

clicked_points = []
drawing_done = False

def mouse_callback(event, x, y, flags, param):
    global clicked_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_done:
        clicked_points.append([x, y])
        if len(clicked_points) == 3:
            drawing_done = True

def calculate_joint_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle_rad = np.arccos(np.clip(np.dot(v1, v2) /
                                  (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return 180 - angle_deg

cap = cv2.VideoCapture("path/to/your/video") 
ret, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 0)
#image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

cv2.imshow("Select 3 Points", first_frame)
cv2.setMouseCallback("Select 3 Points", mouse_callback)

while not drawing_done:
    temp_frame = first_frame.copy()
    for pt in clicked_points:
        cv2.circle(temp_frame, tuple(pt), 5, (0, 255, 0), -1)
    cv2.imshow("Select 3 Points", temp_frame)
    if cv2.waitKey(1) == 27:  
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select 3 Points")

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_pts = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    if status.sum() == 3:
        a, b, c = [pt.ravel() for pt in next_pts]
        cv2.circle(frame, tuple(a.astype(int)), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(b.astype(int)), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(c.astype(int)), 5, (0, 0, 255), -1)
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 0), 2)
        cv2.line(frame, tuple(c.astype(int)), tuple(b.astype(int)), (255, 255, 0), 2)

        angle = calculate_joint_angle(a, b, c)
        cv2.putText(frame, f"{angle:.2f} deg", tuple(b.astype(int) + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    prev_gray = gray.copy()
    prev_pts = next_pts.copy()

    cv2.imshow("Tracked Video", frame)
    if cv2.waitKey(30) == 27:  
        break

cap.release()
cv2.destroyAllWindows()