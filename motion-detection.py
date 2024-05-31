import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create the background subtractor with tweaked parameters
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtractor
    fgmask = fgbg.apply(frame)
    
    # Use Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(fgmask, (11, 11), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    
    # Perform morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Increase the minimum contour area to filter out small movements
        if cv2.contourArea(contour) < 1500:
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Baby Cradle Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()