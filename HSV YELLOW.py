import cv2
import pyautogui
import numpy as np

# Disable fail-safe
pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Define the lower and upper bounds of yellow color in HSV
# These values are set for bright yellow, but you might need to adjust them for your stylus
lower_bound = np.array([20, 100, 100])  # Lower bound for yellow in HSV
upper_bound = np.array([40, 255, 255])  # Upper bound for yellow in HSV

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space (easier for color detection)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask the frame to detect the yellow stylus
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours of the detected object (stylus)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the stylus)
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center of the bounding box (stylus position)
        stylus_x = x + w // 2
        stylus_y = y + h // 2

        # Scale the coordinates to the screen size
        screen_x = int((stylus_x / frame.shape[1]) * screen_width)
        screen_y = int((stylus_y / frame.shape[0]) * screen_height)

        # Move the cursor smoothly to the calculated position
        pyautogui.moveTo(screen_x, screen_y, duration=0.05)

    # Show the mask and the original frame
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
 