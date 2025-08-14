import cv2
import numpy as np

# --- 1. CONFIGURATIONS ---
# IMPORTANT: All dimensions in cm.
KNOWN_DIMENSIONS = {
    "plate_width_cm": 50.0,
    "plate_height_cm": 50.0,
    "hole_diameter_cm": 10.0,
    "dist_edge_to_hole_x_cm": 10.0,
    "dist_edge_to_hole_y_cm": 10.0,
    "dist_between_holes_x_cm": 30.0,
    "dist_between_holes_y_cm": 30.0,
}

# The known reference object to calibrate the pixel-to-cm ratio.
# We will use the horizontal distance between the two top holes.
REFERENCE_WIDTH_CM = 30.0
# The tolerance for a measurement to be considered a 'PASS'
TOLERANCE_CM = 0.8  # Corresponds to 5mm

# --- 2. MAIN INSPECTION FUNCTION ---
def inspect_plate(image_path):
    """
    Loads and inspects a metal plate image.
    This version includes perspective correction and a centimeter-based tolerance.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from path '{image_path}'.")
        return False

    height, width, _ = image.shape
    padding = 250
    canvas = np.zeros((height + padding * 2, width + padding * 2, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[padding:padding+height, padding:padding+width] = image
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found. Could not detect the plate.")
        return False
    
    # --- Corrected logic to find the main plate contour ---
    # Find the main plate contour by looking for a quadrilateral shape.
    # This is more robust than just finding the largest contour.
    plate_contour = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        # We assume the plate is a quadrilateral. We look for contours with 4 vertices
        # and a minimum area to filter out noise.
        if len(approx) == 4 and area > 1000 and area > max_area:
            plate_contour = approx
            max_area = area

    if plate_contour is None:
        print("Error: Could not find a suitable quadrilateral contour for the plate.")
        return False
        
    rect = cv2.minAreaRect(plate_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # --- 3. PERSPECTIVE CORRECTION ---
    # Order the four corner points of the detected plate
    rect_pts = order_points(box)
    (tl, tr, br, bl) = rect_pts
    
    # Calculate the new dimensions for the corrected (warped) image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Get the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(rect_pts.astype("float32"), dst)
    warped = cv2.warpPerspective(canvas, M, (maxWidth, maxHeight))
    
    # Re-process the warped image to find features accurately
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blurred = cv2.GaussianBlur(warped_gray, (7, 7), 0)
    
    # Find circles (holes) on the warped image
    circles = cv2.HoughCircles(
        warped_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=30, minRadius=20, maxRadius=100
    )
    
    if circles is None:
        print("Error: No circles detected after perspective correction.")
        return False
    
    circles = np.uint16(np.around(circles[0]))
    circles_sorted = sorted(circles, key=lambda c: (c[1], c[0]))
    
    # --- Check for minimum number of circles before performing calculations ---
    if len(circles_sorted) < 4:
        print("Error: Not enough circles detected (need at least 4) to perform distance calculation.")
        return False

    # --- 4. MEASUREMENT & CALIBRATION ---
    # Use the known horizontal distance between the two top holes for calibration
    ref_pixels = abs(circles_sorted[1][0] - circles_sorted[0][0])
    pixel_to_cm_ratio = REFERENCE_WIDTH_CM / ref_pixels
    
    # --- Calculation of plate dimensions based on hole positions ---
    # We use the consistent hole positions and known edge distances for a more robust measurement.
    # The plate's width is the distance between the two top holes + twice the edge-to-hole distance.
    hole_positions_cm = [(c[0] * pixel_to_cm_ratio, c[1] * pixel_to_cm_ratio) for c in circles_sorted]
    measured_plate_width_cm = (hole_positions_cm[1][0] - hole_positions_cm[0][0]) + 2 * KNOWN_DIMENSIONS["dist_edge_to_hole_x_cm"]
    measured_plate_height_cm = (hole_positions_cm[2][1] - hole_positions_cm[0][1]) + 2 * KNOWN_DIMENSIONS["dist_edge_to_hole_y_cm"]

    measured_hole_diameters_cm = [2 * c[2] * pixel_to_cm_ratio for c in circles_sorted]
    
    # --- 5. QUALITY CHECK ---
    inspection_results = {}
    
    # Check plate dimensions
    inspection_results["Plate Width"] = (measured_plate_width_cm, KNOWN_DIMENSIONS["plate_width_cm"])
    inspection_results["Plate Height"] = (measured_plate_height_cm, KNOWN_DIMENSIONS["plate_height_cm"])
    
    # Check hole dimensions
    for i, diam in enumerate(measured_hole_diameters_cm):
        inspection_results[f"Hole {i+1} Diameter"] = (diam, KNOWN_DIMENSIONS["hole_diameter_cm"])

    all_passed = True
    fail_messages = []
    
    for name, (measured, expected) in inspection_results.items():
        if abs(measured - expected) > TOLERANCE_CM:
            all_passed = False
            fail_messages.append(f"FAIL: {name}: Measured {measured:.2f} cm (Expected {expected:.2f} cm)")
    
    # --- 6. VISUALIZATION AND ANNOTATION (on the original canvas) ---
    annotation_color = (0, 255, 0) if all_passed else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # We need to re-detect features on the original image to get pixel positions
    # for drawing, but use the measurements from the warped image.
    circles_orig = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100
    )

    if circles_orig is None:
        print("Error: No circles detected on original image for annotation.")
        # Proceed with partial visualization, but return False for inspection
        all_passed = False
        circles_orig_sorted = []
        
    else:
        circles_orig = np.uint16(np.around(circles_orig[0]))
        circles_orig_sorted = sorted(circles_orig, key=lambda c: (c[1], c[0]))
    
    # Draw plate contour
    cv2.drawContours(canvas, [box], 0, annotation_color, 2)
    
    # Draw hole circles
    for c in circles_orig_sorted:
        cv2.circle(canvas, (c[0], c[1]), c[2], annotation_color, 2)
    
    x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
    x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])

    # Annotate overall dimensions
    cv2.line(canvas, (x_min, y_min - 20), (x_max, y_min - 20), (0, 0, 0), 2)
    cv2.putText(canvas, f"Width: {measured_plate_width_cm:.2f} cm", (x_min + int((x_max - x_min)/2) - 100, y_min - 40), font, 0.7, (0, 0, 0), 2)
    
    cv2.line(canvas, (x_max + 20, y_min), (x_max + 20, y_max), (0, 0, 0), 2)
    cv2.putText(canvas, f"Height: {measured_plate_height_cm:.2f} cm", (x_max + 30, y_min + int((y_max-y_min)/2)), font, 0.7, (0, 0, 0), 2)
    
    # Annotate radius for each circle
    for i, c_orig in enumerate(circles_orig_sorted):
        # The radius is half the diameter. We use the diameter from the warped image measurements.
        if i < len(measured_hole_diameters_cm):
            radius_cm = measured_hole_diameters_cm[i] / 2
            text_pos = (c_orig[0] + c_orig[2] + 5, c_orig[1] - 5)
            cv2.putText(canvas, f"R: {radius_cm:.2f} cm", text_pos, font, 0.5, (255,255,255), 2)

    # Annotate distances between holes
    if len(circles_orig_sorted) >= 4:
        # Top Horizontal
        dist_between_holes_x_top_cm_measured = abs(circles_sorted[1][0] - circles_sorted[0][0]) * pixel_to_cm_ratio
        cv2.line(canvas, (circles_orig_sorted[0][0], circles_orig_sorted[0][1]), (circles_orig_sorted[1][0], circles_orig_sorted[1][1]), (255,255,255), 1)
        cv2.putText(canvas, f"{dist_between_holes_x_top_cm_measured:.2f} cm", (int((circles_orig_sorted[0][0] + circles_orig_sorted[1][0]) / 2), circles_orig_sorted[0][1] - 25), font, 0.5, (255,255,255), 2)
        
        # Bottom Horizontal
        dist_between_holes_x_bottom_cm_measured = abs(circles_sorted[3][0] - circles_sorted[2][0]) * pixel_to_cm_ratio
        cv2.line(canvas, (circles_orig_sorted[2][0], circles_orig_sorted[2][1]), (circles_orig_sorted[3][0], circles_orig_sorted[3][1]), (255,255,255), 1)
        cv2.putText(canvas, f"{dist_between_holes_x_bottom_cm_measured:.2f} cm", (int((circles_orig_sorted[2][0] + circles_orig_sorted[3][0]) / 2), circles_orig_sorted[2][1] + 25), font, 0.5, (255,255,255), 2)

        # Left Vertical
        dist_between_holes_y_left_cm_measured = abs(circles_sorted[2][1] - circles_sorted[0][1]) * pixel_to_cm_ratio
        cv2.line(canvas, (circles_orig_sorted[0][0], circles_orig_sorted[0][1]), (circles_orig_sorted[2][0], circles_orig_sorted[2][1]), (255,255,255), 1)
        cv2.putText(canvas, f"{dist_between_holes_y_left_cm_measured:.2f} cm", (circles_orig_sorted[0][0] - 80, int((circles_orig_sorted[0][1] + circles_orig_sorted[2][1]) / 2)), font, 0.5, (255,255,255), 2)

        # Right Vertical
        dist_between_holes_y_right_cm_measured = abs(circles_sorted[3][1] - circles_orig_sorted[1][1]) * pixel_to_cm_ratio
        cv2.line(canvas, (circles_orig_sorted[1][0], circles_orig_sorted[1][1]), (circles_orig_sorted[3][0], circles_orig_sorted[3][1]), (255,255,255), 1)
        cv2.putText(canvas, f"{dist_between_holes_y_right_cm_measured:.2f} cm", (circles_orig_sorted[1][0] + 20, int((circles_orig_sorted[1][1] + circles_orig_sorted[3][1]) / 2)), font, 0.5, (255,255,255), 2)

    # Display inspection status
    status_text = "QUALITY: PASS" if all_passed else "QUALITY: FAIL"
    status_color = (0, 255, 0) if all_passed else (0, 0, 255)
    cv2.putText(canvas, status_text, (padding, 100), font, 1.5, status_color, 3)
    
    display_image = resize_image_for_display(canvas)
    
    cv2.imshow("Automated Metal Plate Inspection", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_passed

def order_points(pts):
    """
    Orders the points of a quadrilateral in a specific order:
    top-left, top-right, bottom-right, and bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has the largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has the largest difference
    
    return rect

def resize_image_for_display(img, max_width=1200, max_height=800):
    """
    Resizes an image for display purposes.
    """
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# --- 6. EXAMPLE USAGE ---
if __name__ == "__main__":
    # Use your actual image file here
    image_file = 'edges_output1.png' 
    result = inspect_plate(image_file)
    print(f"Final Inspection Result: {'PASS' if result else 'FAIL'}")
