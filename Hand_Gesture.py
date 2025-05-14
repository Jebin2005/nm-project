import numpy as np
import cv2
import math

# Load video file instead of webcam
capture = cv2.VideoCapture('input.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break  # End of video

    # Resize for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Use entire frame for detection
    crop_image = frame.copy()

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Skin color range for mask
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Morphological transformations to filter noise
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros(crop_image.shape, np.uint8)

    try:
        # Largest contour
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Convex hull and defects
        hull = cv2.convexHull(contour)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)

        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 2)

        count_defects = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.dist(end, start)
                b = math.dist(far, start)
                c = math.dist(end, far)
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / math.pi)

                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image, far, 5, [0, 0, 255], -1)

                cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Display gesture text
        text = ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
        if count_defects < 5:
            cv2.putText(frame, text[count_defects], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    except:
        pass

    # Show the results
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow("Gesture", frame)
    cv2.imshow("Contours", all_image)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
