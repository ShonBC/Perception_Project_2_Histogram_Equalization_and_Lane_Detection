import glob
import cv2

def img_stitch(folder_path):
    
    for img in glob.glob(folder_path + "*.png"):
        frame = cv2.imread(img)


if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('projection.mp4', fourcc, 20.0, (1920, 1080))