import cv2
import mediapipe as mp
import pyautogui

# Disable fail-safe
pyautogui.FAILSAFE = False

# Initialize Mediapipe Hands and PyAutoGUI
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Decrease screen size (scale the screen size down)
scale_factor = 1 # Decrease this to shrink the screen size (e.g., 0.5 means half the original size)
new_screen_width = int(screen_width * scale_factor)
new_screen_height = int(screen_height * scale_factor)

# Open webcam
cap = cv2.VideoCapture(0)

# Define an offset to position the cursor above the index finger
y_offset = -50  # Adjust this value to place the cursor above the finger (increase or decrease based on need)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Reverse the x-axis (mirroring correction)
            screen_x = int((1 - index_finger_tip.x) * screen_width)

            # Convert coordinates to screen space (x and y) relative to screen resolution and scale down to the new screen size
            screen_x = int(screen_x * scale_factor)
            screen_y = int(index_finger_tip.y * screen_height * scale_factor) + y_offset  # Apply the offset to place cursor above the finger

            # Move the cursor smoothly to the calculated position
            pyautogui.moveTo(screen_x, screen_y, duration=0.9)

            # Check if thumb and index finger are close enough to simulate a click
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            if distance < 0.02:
                pyautogui.click()

    # Show the frame with hand landmarks
    cv2.imshow("Hand Gesture Mouse Control", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
