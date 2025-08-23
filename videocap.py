import cv2
import numpy as np

# --- 1. Define Aruco Marker and Quality Control Parameters ---
# Define the dictionary and the known size of your Aruco marker in cm
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
marker_size_cm = 5.37

# Define the true dimensions of the object in centimeters for quality check
true_length_cm = 7.96
true_breadth_cm = 3.98
true_hole_diameter_cm = 0.54

# Define the tolerance in centimeters
tolerance_cm = 0.1

# --- 2. Start Live Video Capture ---
# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Live video started. Press 's' to capture a frame for analysis, or 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    # Display the live video feed
    cv2.imshow("Live Video Feed", frame)

    # --- Keypress Event Handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Press 'q' to quit the program
        break
    elif key == ord('s'):
        # Press 's' to capture a frame and perform analysis
        
        # Make a copy of the current frame to analyze
        captured_frame = frame.copy()
        
        # --- Start Analysis on the Captured Frame ---
        output = captured_frame.copy()

        # Aruco Marker Detection for Calibration
        corners, ids, _ = cv2.aruco.detectMarkers(captured_frame, aruco_dict, parameters=aruco_params)

        pixels_per_cm = 0
        if ids is not None:
            # Get the length of the first detected marker's side in pixels
            side_length_pixels = np.sqrt(np.sum((corners[0][0][0] - corners[0][0][1]) ** 2))
            pixels_per_cm = side_length_pixels / marker_size_cm
            print(f"\nAuto-calibrated: {pixels_per_cm:.2f} pixels per cm")
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
        else:
            print("\nError: Aruco marker not found in captured frame. Cannot calibrate scale.")
            cv2.imshow("Analysis Result", output)
            cv2.waitKey(0)
            cv2.destroyWindow("Analysis Result")
            continue # Continue the loop to wait for another 's' keypress

        # Object Dimension Detection
        gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No object found in captured frame.")
            cv2.imshow("Analysis Result", output)
            cv2.waitKey(0)
            cv2.destroyWindow("Analysis Result")
            continue

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        length_cm = h / pixels_per_cm
        breadth_cm = w / pixels_per_cm

        # Draw bounding box and text
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f"Length: {length_cm:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(output, f"Breadth: {breadth_cm:.2f} cm", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Hole Detection
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        hole = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=200, param2=20, minRadius=0, maxRadius=0)

        quality_status = "PASS"
        status_color = (0, 255, 0)
        
        # Quality Check Logic
        if not (true_length_cm - tolerance_cm <= length_cm <= true_length_cm + tolerance_cm):
            quality_status = "FAIL"
            status_color = (0, 0, 255)
        if not (true_breadth_cm - tolerance_cm <= breadth_cm <= true_breadth_cm + tolerance_cm):
            quality_status = "FAIL"
            status_color = (0, 0, 255)

        if hole is not None:
            hole = np.uint16(np.around(hole))
            for i in hole[0, :]:
                diameter_cm = (i[2] * 2) / pixels_per_cm
                if not (true_hole_diameter_cm - tolerance_cm <= diameter_cm <= true_hole_diameter_cm + tolerance_cm):
                    quality_status = "FAIL"
                    status_color = (0, 0, 255)
                
                # Draw hole and diameter
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.putText(output, f"Diameter: {diameter_cm:.2f} cm", (i[0] - 50, i[1] + i[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            print("No circles found.")

        # Put the final quality status text
        cv2.putText(output, f"Quality: {quality_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Display the analyzed frame
        cv2.imshow("Analysis Result", output)
        cv2.waitKey(0) # Wait until a key is pressed to close the analysis window
        cv2.destroyWindow("Analysis Result") # Close the analysis window
        
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
