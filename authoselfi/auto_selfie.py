import cv2
import time
import pygame

# Initialize pygame mixer (used for loading and playing audio)
pygame.mixer.init()

# Load sounds (Load the sound files for the "ready" and "capture" sounds.)
ready_sound = pygame.mixer.Sound('audio\get_ready_sound.mp3')
capture_sound = pygame.mixer.Sound('audio/camera sound.mp3')

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Counter for saved images
img_counter = 0
last_capture_time = 0
capture_interval = 7  # Interval between captures in seconds

ready_announced = False

# Function to detect and draw on faces, eyes, and smiles
def detect_and_draw(frame, original_frame):
    global img_counter, last_capture_time, ready_announced
    current_time = time.time()

    # Initialize time_since_last_capture
    time_since_last_capture = current_time - last_capture_time
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # Reset the countdown if no face is detected
        last_capture_time = current_time
        ready_announced = False

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # Update time_since_last_capture
            time_since_last_capture = current_time - last_capture_time

            # Check if it's time to capture the photo
            time_remaining = capture_interval - time_since_last_capture

            # Announce "Ready" 5 seconds before capture
            if time_remaining <= 3 and not ready_announced:
                ready_sound.play()
                ready_announced = True

            # Take a selfie if a smile is detected and the time interval has passed
            if time_remaining <= 0:
                img_name = f'selfie_{img_counter}.png'
                cv2.imwrite(img_name, original_frame)
                capture_sound.play()
                img_counter += 1
                last_capture_time = current_time
                ready_announced = False
                print(f"Smile detected! Photo captured and saved as {img_name}.")
                return  # Exit the loop after capturing one photo

    # If no smile is detected after "get ready" sound has played, reset the countdown
    if ready_announced and time_since_last_capture > capture_interval:
        last_capture_time = current_time
        ready_announced = False
        print("No smile detected. Resetting countdown.")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the camera feed
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)  # Allow window to be resizable

# Display the camera feed window and wait for 2 seconds
ret, frame = cap.read()
cv2.imshow('Camera Feed', frame)
cv2.waitKey(1)
time.sleep(1)  # Delay for 1 second before starting the countdown

# Set the initial capture time after the delay
last_capture_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Create a copy of the original frame to save without drawings
    original_frame = frame.copy()
    
    # Perform face, eye, and smile detection and draw on the frame
    detect_and_draw(frame, original_frame)
    
    # Display the camera feed in the window
    cv2.imshow('Camera Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
