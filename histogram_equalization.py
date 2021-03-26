import cv2
import numpy as np

cap = cv2.VideoCapture('media/NightDrive-2689.mp4')


def pmf(frame): # Calculate Probability Mass Function (PMF)
   
    row = len(frame)
    col = len(frame[0])
    pixels = row * col # Total pixel count

    count = [0] * 256 # Store the count of each pixel value (0-255)

    for i in range(len(frame)):
        for j in range(len(frame[i])):

            count[frame[i][j]] += 1 

    PMF = np.array(count) 
    PMF = PMF / pixels # Store PMF percentage

    return PMF      

def cdf(PMF): # Calculate the Cumulative Distribution Function (CDF)

    CDF = [0]

    # while i < 255:
        # for j in range(len(PMF)):
        
        #     PMF[i]
        #     CDF.append(PMF[i] + CDF[-1])
        #     i = CDF[-1] # Set i to be the value of the last index in CDF    

    return CDF          

if __name__ == "__main__":

    # Define Video out properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('projection.mp4', fourcc, 20.0, (1920, 1080))
        
    while True:
        ret, frame = cap.read()

        """
        Break frame out into the 3 color channels
        Calculate PMF
        Calculate CDF
        Multiply each pixel with its corrisponding CDF value for each channel
        Combine the color channels
        Display the frame

        """

        b, g, r = cv2.split(frame) # Separate frame into 3 color channels

        # Calculate probability mass function
        b_PMF = pmf(b)
        g_PMF = pmf(g)
        r_PMF = pmf(r)

        # Calculate Cumulative Distribution Function (CDF)
        b_CDF = cdf(b_PMF)
        g_CDF = cdf(g_PMF)
        r_CDF = cdf(r_PMF)
        

        cv2.imshow("Video Feed", frame) # Show the projection

        # Condition to break the while loop
        i = cv2.waitKey(50)
        if i == 27:
            break    

    # Wait for ESC key to be pressed before releasing capture method and closing windows
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()