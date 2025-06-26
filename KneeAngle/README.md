# A Simple Program That Helped Me Measure Knee Flexion Accurately During Post-Surgery Recovery
### Used OpenCV, NumPy and math to handle consecutive click events on the image from the user, transform them into vectors and calculate the respective joint angle

This tool proved highly valuable during my post-operative rehabilitation. Estimating the knee joint angle by eye is imprecise, and this software provided a reliable way to measure the **exact degree of flexion**. It helped me monitor the stability and safe range of motion in my knee, ensuring I could bend it to the appropriate extent without risking further injury.<br>
In addition to the still-image version of the program, I have implemented a video-based extension that tracks the three selected points across frames using the **Lucas-Kanade Optical Flow** method. This live tracking version is designed for relatively static video footage and performs best under consistent lighting conditions. Despite providing real-time tracking, the implementation remains lightweight and resource-efficient.

![Knee Angle Estimation](https://github.com/fylexx/Projects/blob/main/KneeAngle/KneeAngleTest.png)
