# Perception_project2

# ENPM 673 - Perception for Autonomous Robots:

# Project 2 - Histogram Equalization and Lane Detection

# Shon Cortes

histogram_equalization.py:

Packages Used:

    1. import numpy
    2. import cv2 

Run the program:

        1. Run the program using your preferred method (terminal...)

        Example in Terminal:

            python3 histogram_equalization.py
        
        Output will show the original video, the gray scale equalization, the color equalization, as well as gamma correction videos.
        
    Program Summary:
        
        Use histogram equalization techniques to enhance the contrast and improve the visual appearance of a video.
        The program will show histogram equalization applied to a gray scale video as well as the color video (by splitting the video into its 3 color channels, using histogram equalization, then combining the color channels), as well as a gamma correction technique to adjust contrast.
        
        
lane_detection_data_1.py:

Packages Used:

    1. import numpy
    2. import cv2 
    3. import glob


Run the program:

        1. Run the program using your preferred method (terminal...)

        Example in Terminal:

            python3 lane_detection_data_1.py
        
        Output will show the original video with the detected lanes filled in.
        
    Program Summary:
        
        Use hough transform for lane detection to mimic Warning systems used in Self Driving Cars. 
        The program will do canny edge detection then isolate a region of interest.
        It then detects all lines and sorts them into right and left lanes by checking the slope of the lines and takes their average.
        Finally it overlays the detected lanes on the original video.


lane_detection_data_2.py:

Packages Used:

    1. import numpy
    2. import cv2 


Run the program:

        1. Run the program using your preferred method (terminal...)

        Example in Terminal:

            python3 lane_detection_data_2.py
        
        Output will show the original video with the detected lanes filled in.
        
    Program Summary:
        
        Use hough transform for lane detection to mimic Warning systems used in Self Driving Cars. 
        The program will use homography to isolate a region of interest and filter the frame for only white and yellow pixels.
        It then does canny edge detection and further isolates a region of interest.
        It then detects all lines in the yellow and white filtered frames and takes their average.
        Finally it overlays the detected lanes on the original video. If no lines are detected it used the lines from the previous frame.
