"""
Shon Cortes
ENPM 673 - Perception for Autonomous Robots:
Project 2 Lane Detection on Data Set 2
"""

import cv2
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

    h, w, _ = img.shape

    src_points = np.float32([[330, 310], [440, 310], [110,460], [625, 460]])
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M, mask = cv2.findHomography(src_points, dst_points)
    M_inv, mask = cv2.findHomography(dst_points, src_points)
    # M = cv2.getPerspectiveTransform(src_points, dst_points)

    top = cv2.warpPerspective(img, M, (w, h))
    front = cv2.warpPerspective(img, M_inv, (width, height))

    return top, front

def color_mask(frame): #  Return a mask of only the detected yellow and white pixels.
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_min = np.array([14, 25, 100])
    yellow_max = np.array([100, 255, 255])

    ymask = cv2.inRange(hsv, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200])
    white_max =np.array([255, 255, 255])

    wmask = cv2.inRange(hsv, white_min, white_max)

    return ymask, wmask 

def fit(frame, y_lines, w_lines): # Takes the average of the lines detected and creates a best fit line 

    global ylines, wlines

    if y_lines is None:
        y_lines = ylines
        yline_avg = np.average(y_lines, axis=0)
        show_lines_avg(frame, yline_avg)
    else:
        ylines = y_lines
    
        yline_avg = np.average(y_lines, axis=0)
        show_lines_avg(frame, yline_avg)
    
    if w_lines is None:
        w_lines = wlines
        try:
            wline_avg = np.average(w_lines, axis=0)
            show_lines_avg(frame, wline_avg)
        except TypeError:
            
            pass
    
    else:
        wlines = w_lines
        wline_avg = np.average(w_lines, axis=0)
        show_lines_avg(frame, wline_avg)

    try:
        points = np.array([[[wline_avg[0][0], wline_avg[0][1]], [wline_avg[0][2], wline_avg[0][3]], [yline_avg[0][0], yline_avg[0][1]], [yline_avg[0][2], yline_avg[0][3]]]], dtype=np.int32)
        cv2.fillPoly(frame, points, (0,0,255))
    except IndexError:
            pass

def line_pos(line): # Using line slope and y-intercept, return x,y coordinates

    slope = line[0]
    y_int = line[1]

    x1 = int((height - y_int) / slope)
    y1 = height
    y2 = int(height / 2)
    x2 = int((y2 - y_int) / slope)
    
    ln_coord = np.array([x1, y1, x2, y2])

    return ln_coord    

def h_lines(roi): # Use Probabilistic Hough Transform to detect lines from canny edges 

    rho_tolerance = 2 # measured in pixels
    theta_tolerance = np.pi / 180 # measured in radians
    min_vote = 100 # minimum points on the line detected for it to be considered as a line
    min_length = 20

    lines = cv2.HoughLinesP(roi, rho_tolerance, theta_tolerance, min_vote, minLineLength= min_length, maxLineGap= 5000)

    return lines

def show_lines(frame, lines): # Display lines

    if lines is not None: 
            for i in range(len(lines)):
                l = lines[i][0]
                cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3) 

def show_lines_avg(frame, lines): # Display lines

    if lines is not None: 
         for i in range(len(lines)):
                l = lines[i]   
                cv2.line(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 3) 

def cnts(image): # Use canny edge detection to define and display edges. Returns AR Tag x, y coordinates.

    k = 9
    blurr = cv2.GaussianBlur(image, (k, k), 0) # Blur frame

    threshold_1 = 10 # Define Canny edge detection thresholds
    threshold_2 = 100 # Define Canny edge detection thresholds
    canny = cv2.Canny(blurr, threshold_1, threshold_2) # Call Canny edge detection on blurred image
    
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return canny, contours

def ROI(frame): # Isolate the region of interest (the road)

    roi = np.array([[(0, height / 2), (0, height), (width, height), (width, height / 2)]], dtype= np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi, 255)

    mask = cv2.bitwise_and(frame, mask)

    return mask

def fit_poly(contours, x_start, x_lim):

    x_cord = []
    y_cord = []

    for i in range(len(contours)):
        for j in range(len(contours[i])):

            x_cord.append(contours[i][j][0][0])
            y_cord.append(contours[i][j][0][1])

    try:

        curve = np.polyfit(x_cord, y_cord, 2)
        # x_range = np.linspace(0, 500, 100) top.shape[1] / 2
        x_range = np.linspace(start= x_start, stop= x_lim, num= 100)
        y_range = np.polyval(curve, x_range)

        drawl = (np.asarray([x_range, y_range]).T).astype(np.int32)
    
    except TypeError:
        return

    return drawl

def func(x, a, b, c):
    return a * x + b*x**2 + c

def draw_poly(frame, draw):

    cv2.polylines(frame, [draw], False, (255, 0, 0), 3)

if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lane_detection_2.mp4', fourcc, 240, (720,480))

    cap = cv2.VideoCapture('media/data_2/challenge_video.mp4')

    ylines = []
    wlines = []
    lines_ = []  
    yline_avg = []
    wline_avg = []

    while True:
         
        ret, frame = cap.read()

        if not ret: # If no frame returned, break the loop
            break
        
        frame = cv2.resize(frame, (720,480))
        height, width, _ = frame.shape
        frame = undistort(frame) # Undistort frame with camera matrix
        top, _ = homog(frame)
        _, front = homog(top)

        ymask, wmask = color_mask(front)
        ycanny, ycontours = cnts(ymask)
        wcanny, wcontours = cnts(wmask)

        # Isolate Reigon of Interest
        y_roi = ROI(ycanny)
        w_roi = ROI(wcanny)

        # Find lines using Hough Lines
        y_lines = h_lines(ycanny)
        w_lines = h_lines(wcanny)
       
       # Take average of lines found, group into left and right lines, show lines and fill lane
        fit(frame, y_lines, w_lines)

        cv2.imshow('frame', frame)
        
        # out.write(frame) # Uncomment to save new output of desired frame
        
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