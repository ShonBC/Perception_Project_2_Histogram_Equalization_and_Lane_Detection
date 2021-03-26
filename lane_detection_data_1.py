import glob
import cv2

def img_stitch():
    
    for img in glob.glob("media/data_1/data/*.png"):
        frame = cv2.imread(img)
        out.write(frame)




if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lane_detection_1.mp4', fourcc, 240, (1920, 1080))