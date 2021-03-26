import cv2


if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('lane_detection_2.mp4', fourcc, 240, (1920, 1080))