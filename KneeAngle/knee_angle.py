'''
Program to measure the angle of a bend knee based on user input.
INPUTS: User needs to click three points on the image, preferrably shin, knee and upper thigh.
OUTPUTS: Calculates angle of vectors between the three points.
'''
import cv2
import numpy as np
import math

points = []
angles = []

def calculate_angle(p1, p2, p3):
    global angles
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    angle_rad = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    joint_deg = 180 - angle_deg
    angles.append(joint_deg)

    return joint_deg

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        if len(points) == 3:
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)
            cv2.line(img, points[1], points[2], (0, 0, 255), 2)
            angle = calculate_angle(points[0], points[1], points[2])
            print(f"Angle at point {points[1]}: {angle:.2f} degrees")
            cv2.putText(img, f"{angle:.2f} deg", points[1], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            points= []
            mean_angle = np.mean(np.array(angles))
            print(f"Mean Angle over {len(angles) * 3} points: {mean_angle}")
            #cv2.putText(img, f"Mean Angle over {len(angles) * 3} points: {mean_angle}", (1,84), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)
        cv2.imshow("Image", img)

img = cv2.imread("/path/to/your/image")  
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
