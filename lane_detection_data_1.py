"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 2 Lane Detection on Data Set 1 
"""
import glob
import cv2
from cv2 import data
import copy
import numpy as np

def img_stitch(): # Takes images from data folder and creates a video file to be used for lane detection
    
    video = []
    images = []

    for filename in glob.glob("media/data_1/data/*.png"): # Creates array of all images to be used in video file
        
        images.append(filename)

    images.sort()

    for img in images: # Creates array of all images to be used in video file
            frame = cv2.imread(img)
            video.append(frame)
            
            height, width, _ = frame.shape
            size = (width, height)

    

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    data_out = cv2.VideoWriter('data_1.mp4', fourcc, 20, size)

    for i in range(len(video)): # Writes frames to video file

        data_out.write(video[i])
    data_out.release

def cnts(image): # Use canny edge detection to define and display edges. Returns AR Tag x, y coordinates.

    k = 5
    blurr = cv2.GaussianBlur(image, (k, k), 0) # Blurr frame

    threshold_1 = 50 # Define Canny edge detection thresholds
    threshold_2 = 150 # Define Canny edge detection thresholds
    canny = cv2.Canny(blurr, threshold_1, threshold_2) # Call Canny edge detection on blurred image
  
    return canny

def ROI(frame): # Isolate the region of interest (the road)

    roi = np.array([[(55, height - 10), (0, height - 10), (530, height - 10), (340, 200)]], dtype= np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi, 255)

    mask = cv2.bitwise_and(frame, mask)

    return mask

def fit(frame, lines): # Takes the average of the lines detected and creates a best fit line 

    l_lines = []
    r_lines = []

    for i in range(len(lines)):

        x_1, y_1, x_2, y_2 = lines[i].reshape(4)
        param = np.polyfit((x_1, x_2), (y_1, y_2), 1) # Fit to a first degree polynomial

        slope = param[0]
        y_int = param[1]

        if slope < 0:
            l_lines.append((slope,y_int))
        elif int(slope) == 0:
            continue
        else:
            r_lines.append((slope, y_int))
        
    l_avg = np.average(l_lines, axis=0)
    r_avg = np.average(r_lines, axis=0)

    l_bfl = line_pos(l_avg)
    r_bfl = line_pos(r_avg)

    cv2.line(frame, (l_bfl[0], l_bfl[1]), (l_bfl[2], l_bfl[3]), (255, 0, 0), 3)
    cv2.line(frame, (r_bfl[0], r_bfl[1]), (r_bfl[2], r_bfl[3]), (255, 0, 0), 3)

    points = np.array([[[r_bfl[0], r_bfl[1]], [l_bfl[0], l_bfl[1]], [l_bfl[2], l_bfl[3]], [r_bfl[2], r_bfl[3]]]], dtype=np.int32)
    cv2.fillPoly(frame, points, (0,0,255))

    return l_bfl, r_bfl    

def line_pos(line): # Using line slope and y-intercept, return x,y coordinates

    slope = line[0]
    y_int = line[1]

    x1 = int((height - y_int) / slope)
    y1 = height
    y2 = int(height / 2)
    x2 = int((y2 - y_int) / slope)
    
    ln_coord = np.array([x1, y1, x2, y2])

    return ln_coord    

def h_lines(frame, roi): # Use Probablistic Hough Transform to detect lines from canny edges 

    rho_tolerance = 2 # measured in pixels
    theta_tolerance = np.pi / 180 # measured in radians
    min_vote = 100 # minimum points on the line detected for it to be considered as a line
    min_length = 100

    lines = cv2.HoughLinesP(roi, rho_tolerance, theta_tolerance, min_vote, minLineLength= min_length, maxLineGap= 80)

    return lines

def show_lines(frame, lines): # Display lines

    if lines is not None: 
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3)    
    
def top_down(img): # Homography for top down view

    w, h = 200, 500

    # src_points = np.float32([[340, 215], [350, 215], [109, 480], [445, 480]])
    src_points = np.float32([[285, 240], [365, 250], [0, 380], [450, 450]])
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M, mask = cv2.findHomography(src_points, dst_points)
    # M = cv2.getPerspectiveTransform(src_points, dst_points)

    top = cv2.warpPerspective(img, M, (w, h))

    return top

if __name__ == "__main__":

    # Output video parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lanes_data_1.mp4', fourcc, 20, (720,480))
    
    # img_stitch() # Stitch images together for a video file
    cap = cv2.VideoCapture('media/data_1/data_1.mp4')

    while True:

        ret, frame = cap.read()
        
        if not ret: # If no frame returned, break the loop  
            break

        frame = cv2.resize(frame, (720,480))
        height, width, _ = frame.shape

        canny = cnts(frame) # Edge detection
        roi = ROI(canny) # Region of Interest
        lines = h_lines(frame, roi) # Compute Hough lines

        fit(frame, lines) # Average the returned Hough lines

        # cv2.imshow('ROI', roi)

        cv2.imshow("frame", frame) # Show resulting frame with lanes drawn on
        out.write(frame)       

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()