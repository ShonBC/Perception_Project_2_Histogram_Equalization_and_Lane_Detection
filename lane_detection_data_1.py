import glob
import cv2
from cv2 import data

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




if __name__ == "__main__":

    # Define Video out properties
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('lane_detection_1.mp4', fourcc, 144, (1920, 1080))

    # img_stitch() # Stitch images together for a video file
    cap = cv2.VideoCapture('media/data_1/data_1.mp4')

    while True:

        ret, frame = cap.read()

        cv2.imshow("frame", frame)
        # Condition to break the while loop
        # i = cv2.waitKey(50)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()