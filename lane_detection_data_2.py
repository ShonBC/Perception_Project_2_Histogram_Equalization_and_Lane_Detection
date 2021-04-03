from typing import SupportsFloat
import cv2
import lane_detection_data_1
import numpy as np


def undistort(frame): # Undistort frame with camera calibration matrix and distortion matrix

    #Camera Matrix
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
    [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    #Distortion Coefficients
    dist = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

    # dis_frame = cv2.undistort(frame, np.float32(K), np.float32(dist), None)
    dis_frame = cv2.undistort(frame, K, dist, None)

    return dis_frame

def homog(img): # Homography for top down view and reverting back to front view

    w, h = 200, 500

    src_points = np.float32([[330, 310], [440, 310], [110,460], [625, 460]])
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M, mask = cv2.findHomography(src_points, dst_points)
    M_inv, mask = cv2.findHomography(dst_points, src_points)
    # M = cv2.getPerspectiveTransform(src_points, dst_points)

    top = cv2.warpPerspective(img, M, (w, h))
    front = cv2.warpPerspective(img, M_inv, (width, height))

    return top, front

def color_mask(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_min = np.array([20, 15, 20])
    yellow_max = np.array([30, 255, 255])

    ymask = cv2.inRange(hsv, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200])
    white_max =np.array([255, 255, 255])

    wmask = cv2.inRange(hsv, white_min, white_max)

    return ymask, wmask


def fit(frame, lines): # Takes the average of the lines detected and creates a best fit line 

    l_lines = []
    r_lines = []

    for i in range(len(lines)):

        x_1, y_1, x_2, y_2 = lines[i].reshape(4)
        param = np.polyfit((x_1, x_2), (y_1, y_2), 2) # Fit to a first degree polynomial

        a = param[0]
        b = param[1]
        c = param[2]

        # slope = param[0]
        # y_int = param[1]

        # if slope < 0:
        #     l_lines.append((slope,y_int))
        # elif int(slope) == 0:
        #     continue
        # else:
        #     r_lines.append((slope, y_int))
        
    l_avg = np.average(l_lines, axis=0)
    r_avg = np.average(r_lines, axis=0)

    l_bfl = line_pos(l_avg)
    r_bfl = line_pos(r_avg)

    cv2.line(frame, (l_bfl[0], l_bfl[1]), (l_bfl[2], l_bfl[3]), (255, 0, 0), 3)
    cv2.line(frame, (r_bfl[0], r_bfl[1]), (r_bfl[2], r_bfl[3]), (255, 0, 0), 3)

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
    min_length = 20

    lines = cv2.HoughLinesP(roi, rho_tolerance, theta_tolerance, min_vote, minLineLength= min_length, maxLineGap= 1000)

    return lines

def show_lines(frame, lines): # Display lines

    if lines is not None: 
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3) 

def cnts(image): # Use canny edge detection to define and display edges. Returns AR Tag x, y coordinates.

    k = 3
    blurr = cv2.GaussianBlur(image, (k, k), 0) # Blurr frame

    threshold_1 = 3 # Define Canny edge detection thresholds
    threshold_2 = 50 # Define Canny edge detection thresholds
    canny = cv2.Canny(blurr, threshold_1, threshold_2) # Call Canny edge detection on blurred image
  
    return canny

def ROI(frame): # Isolate the region of interest (the road)

    roi = np.array([[(55, height - 10), (0, height - 10), (530, height - 10), (340, 200)]])

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi, 255)

    mask = cv2.bitwise_and(frame, mask)

    return mask

if __name__ == "__main__":

    # # Define Video out properties
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('lane_detection_2.mp4', fourcc, 240, (1920, 1080))

    cap = cv2.VideoCapture('media/data_2/challenge_video.mp4')

    while True:
         
        ret, frame = cap.read()

        if not ret: # If no frame returned, break the loop
            break
        
        frame = cv2.resize(frame, (720,480))
        height, width, _ = frame.shape
        frame = undistort(frame) # Undistort frame with camera matrix
        top, _ = homog(frame)
        _, front = homog(top)
        cv2.imshow('back', front)

        ymask, wmask = color_mask(top)
        ycanny = cnts(ymask)
        cv2.imshow('ymask', ycanny)
        cv2.imshow('wmask', wmask)

        # top = top_down(frame) # Use Homography for top down view
        cv2.imshow('top', frame)

        # canny = cnts(top) # Edge detection
        # # roi = ROI(canny) # Region of Interest
        # lines = h_lines(frame, canny) # Compute Hough lines
        

        # # try:
        # #     fit(top, lines) # Average the returned Hough lines
        # # except IndexError:
        # #     pass

        # # show_lines(top, lines)
        # cv2.imshow('test', top)
        # cv2.imshow('canny', canny)

        # # fit(frame, top) # Average the returned Hough lines
        
        # cv2.imshow("frame", frame)
        # # out.write(frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

"""
Filter for a region of interest

Try filtering out everything except the lanes by color.

Canny edge detection.

return x,y coordinates of the yellow and white contours. 

fit the coordinates to a 2nd degree polynomial.

draw the lanes.

"""