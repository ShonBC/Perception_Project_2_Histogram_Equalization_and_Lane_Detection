import cv2
import lane_detection_data_1
import numpy as np

#Camera Matrix
K = [[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
 [  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
 [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]

#Distortion Coefficients
dist = [[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
    2.20573263e-02]]

def top_down(img): # Homography for top down view

    w, h = 200, 500

    src_points = np.float32([[320, 320], [440, 320], [160,460], [625, 460]])
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    M, mask = cv2.findHomography(src_points, dst_points)
    # M = cv2.getPerspectiveTransform(src_points, dst_points)

    top = cv2.warpPerspective(img, M, (w, h))

    return top


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
        top = top_down(frame) # Use Homography for top down view
        cv2.imshow('top', top)
        
        cv2.imshow("frame", frame)
        # out.write(frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows( )