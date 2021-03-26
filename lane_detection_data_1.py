import cv2






if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('projection.mp4', fourcc, 20.0, (1920, 1080))