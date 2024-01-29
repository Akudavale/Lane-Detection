import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_state):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = initial_state
        self.estimate_error = 1  # Initial estimate error

    def update(self, measurement):
        # Prediction step
        predicted_state = self.state
        predicted_estimate_error = self.estimate_error + self.process_variance

        # Update step
        kalman_gain = predicted_estimate_error / (predicted_estimate_error + self.measurement_variance)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.estimate_error = (1 - kalman_gain) * predicted_estimate_error

        return self.state

def draw_lines(image, x_values, y_values, color, thickness):
    for i in range(len(y_values)):
        cv2.circle(image, (int(x_values[i]), int(y_values[i])), 2, color, -1)
    for i in range(len(y_values) - 1):
        cv2.line(image, (int(x_values[i]), int(y_values[i])), (int(x_values[i + 1]), int(y_values[i + 1])), color, thickness)

# Step 1: Read and Display an Image
image = cv2.imread('your_image_path.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Image Preprocessing (Convert to Grayscale)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Edge Detection (Canny Edge Detector)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred_image, 50, 150)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Region of Interest (ROI) Selection
height, width = edges.shape
roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
roi_mask = np.zeros_like(edges)
cv2.fillPoly(roi_mask, roi_vertices, 255)
roi_edges = cv2.bitwise_and(edges, roi_mask)
cv2.imshow('Region of Interest', roi_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5: Hough Transform for Line Detection
lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
line_image = np.zeros_like(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

cv2.imshow('Detected Lines', line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 6: Fit a Line to the Detected Lane Markings
left_lane_points = []
right_lane_points = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)

        # Classify lines based on slope (left or right lane)
        if slope < 0:
            left_lane_points.extend([(x1, y1), (x2, y2)])
        else:
            right_lane_points.extend([(x1, y1), (x2, y2)])

# Fit a line to the points using the least squares method
left_lane_line = np.polyfit(*zip(*left_lane_points), 1)
right_lane_line = np.polyfit(*zip(*right_lane_points), 1)

# Initialize Kalman Filters for left and right lanes
kalman_filter_left = KalmanFilter(process_variance=0.1, measurement_variance=10, initial_state=left_lane_line)
kalman_filter_right = KalmanFilter(process_variance=0.1, measurement_variance=10, initial_state=right_lane_line)

# Process each frame and display the result with lane tracking and smoothing
result_image = image.copy()
for i in range(len(lines)):
    # Extract left and right lane points
    left_lane_points, right_lane_points = extract_lane_points(lines[i])

    # Fit lines to the points using the least squares method
    left_lane_line = np.polyfit(*zip(*left_lane_points), 1)
    right_lane_line = np.polyfit(*zip(*right_lane_points), 1)

    # Update Kalman Filters with the new measurements
    kalman_filter_left.update(left_lane_line)
    kalman_filter_right.update(right_lane_line)

    # Get the smoothed estimates from Kalman Filters
    smoothed_left_lane_line = kalman_filter_left.state
    smoothed_right_lane_line = kalman_filter_right.state

    # Generate y values for the smoothed lines
    y_values = np.linspace(0, height, 100)

    # Calculate corresponding x values for the smoothed lines
    smoothed_left_lane_x_values = np.polyval(smoothed_left_lane_line, y_values)
    smoothed_right_lane_x_values = np.polyval(smoothed_right_lane_line, y_values)

    # Draw the detected and smoothed lines on the result image
    draw_lines(result_image, left_lane_x_values, y_values, color=(0, 255, 0), thickness=5)
    draw_lines(result_image, right_lane_x_values, y_values, color=(0, 255, 0), thickness=5)
    draw_lines(result_image, smoothed_left_lane_x_values, y_values, color=(0, 0, 255), thickness=5)
    draw_lines(result_image, smoothed_right_lane_x_values, y_values, color=(0, 0, 255), thickness=5)

cv2.imshow('Lane Detection with Tracking and Smoothing', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
