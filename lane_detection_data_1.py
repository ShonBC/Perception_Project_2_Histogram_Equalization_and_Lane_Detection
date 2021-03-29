# Shon Cortes

import glob
import cv2
from cv2 import data
import copy
import numpy as np

def img_stitch(): # Takes images from data folder and creates a video file to be used for lane detection
    
    video =[]

    for img in glob.glob("media/data_1/data/*.png"): # Creates array of all images to be used in video file
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

    threshold_1 = 10 # Define Canny edge detection thresholds
    threshold_2 = 150 # Define Canny edge detection thresholds
    canny = cv2.Canny(blurr, threshold_1, threshold_2) # Call Canny edge detection on blurred image
    # _, bianary = cv2.threshold(canny, 230, 255, 0) # Convert to bianary image

    # contours, hierarchy = cv2.findContours(bianary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find all contours in image. cv2.RETR_TREE gives all hierarchy information, cv2.CHAIN_APPROX_SIMPLE stores only the minimum required points to define a contour.
        
    # g_cnts = [] # Store only the contours with parents and children.
    # g_hier = [] # Store hierarchy of contours with parents and children.
    
    # for i in range(len(hierarchy)): # Filter out contours with no parent to isolate ar tag.
    #     for j in range(len(hierarchy[i])):
    #         if hierarchy[i][j][3] != -1 and hierarchy[i][j][2] != -1:
    #             g_cnts.append(contours[j])
    #             g_hier.append(hierarchy[i][j])

    # area = [] # Store area of filtered contours with parents and children
    # for i in range(len(g_cnts)):
    #     area.append(cv2.contourArea(g_cnts[i]))

    # cv2.drawContours(image, g_cnts, -1, (0, 255, 0), 3)

    return canny

def ROI(frame): # Isolate the region of interest (the road)

    roi = np.array([[(0, 380), (0, height), (480, height), (340, 200)]])

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi, 255)

    mask = cv2.bitwise_and(frame, mask)

    return mask



def h_lines(frame): # Use Probablistic Hough Transform to detect lines from canny edges 

    rho_tolerance = 1 # measured in pixels
    theta_tolerance = np.pi / 180 # measured in radians
    vote_thresh = 100 # minimum points on the line detected for it to be considered as a line
    min_length = 100

    lines = cv2.HoughLinesP(frame, rho_tolerance, theta_tolerance, vote_thresh, min_length, maxLineGap= 200)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA) 


def top_down(img):

    w, h = 200, 500

    # src_points = np.float32([[340, 215], [350, 215], [109, 480], [445, 480]])
    src_points = np.float32([[285, 240], [365, 250], [0, 380], [450, 450]])
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M, mask = cv2.findHomography(src_points, dst_points)
    # M = cv2.getPerspectiveTransform(src_points, dst_points)

    top = cv2.warpPerspective(img, M, (w, h))

    return top

if __name__ == "__main__":

    # Define Video out properties
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('lane_detection_1.mp4', fourcc, 144, (1920, 1080))

    # img_stitch() # Stitch images together for a video file
    cap = cv2.VideoCapture('media/data_1/data_1.mp4')

    while True:

        ret, frame = cap.read()
        
        if not ret: # If no frame returned, break the loop  
            break

        frame = cv2.resize(frame, (720,480))
        height, width, _ = frame.shape
        # crop = copy.deepcopy(frame[int(height/2 - 30) : height,  :width])
      
        canny = cnts(frame)
        roi = ROI(canny)
        h_lines(roi)

        # cv2.imshow('Canny', canny)
        cv2.imshow('ROI', roi)

        # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # mask = white_mask(crop)
        # canny = cnts(mask)
        # cv2.imshow("white", mask)

        # canny = cnts(crop)
        # h_lines(canny)
        # cv2.imshow("crop", crop)

        cv2.imshow("frame", frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()