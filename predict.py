import os
import sys
import glob
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/digit_model.keras"

# ============================================================
# 1. LOAD MODEL
# ============================================================
def load_digit_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Run train_model.py first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ============================================================
# 2. PREPROCESSING FUNCTIONS
# ============================================================
def preprocess_single_digit_crop(binary_image, x, y, w, h):
    """
    Standard MNIST formatting pipeline:
    1. Crop digit from binary image
    2. Square canvas
    3. Resize to 20x20
    4. Center in 28x28 with 4-pixel border
    5. Return (28, 28, 1) normalized to 0.0-1.0
    """
    cropped = binary_image[y:y+h, x:x+w]
    size = max(w, h)
    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize to 20x20
    digit = cv2.resize(canvas, (20, 20), interpolation=cv2.INTER_AREA)

    # Place in 28x28 MNIST canvas
    mnist_image = np.zeros((28, 28), dtype=np.uint8)
    mnist_image[4:24, 4:24] = digit

    # Normalize and shape for CNN: (28, 28, 1)
    normalized = mnist_image.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=-1)


def detect_and_segment_digits(image_path):
    """
    Detects individual digit bounding boxes from an image.
    Preserves single-digit fallback if only one digit is present.
    Returns:
        binary_image: Binarized inverted image (white digit on black background)
        sorted_boxes: List of (x, y, w, h) bounding boxes sorted left-to-right
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Threshold (standard inverse binary: dark strokes -> white on black background)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. External contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No handwritten digits detected.")

    # 4. Filter contours based on area relative to the largest contour
    max_area = max(cv2.contourArea(c) for c in contours)
    # Filter noise: minimum area threshold
    min_area = max(500.0, max_area * 0.03)

    raw_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise with extreme aspect ratios or tiny dimensions
            if w >= 15 and h >= 25:
                raw_boxes.append((x, y, w, h))

    # Single-digit fallback: if no candidate passed filtering, use largest contour
    if not raw_boxes:
        largest_contour = max(contours, key=cv2.contourArea)
        raw_boxes = [cv2.boundingRect(largest_contour)]

    # 5. Sort left to right
    raw_boxes.sort(key=lambda b: b[0])

    # 6. Merge overlapping bounding boxes horizontally (e.g. disconnected strokes of same digit)
    merged_boxes = []
    for box in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            curr_x, curr_y, curr_w, curr_h = box
            # If current box overlaps significantly horizontally with previous
            if curr_x < prev_x + int(prev_w * 0.65):
                new_x = min(prev_x, curr_x)
                new_y = min(prev_y, curr_y)
                new_w = max(prev_x + prev_w, curr_x + curr_w) - new_x
                new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)

    return binary, merged_boxes


# ============================================================
# 3. PREDICT FUNCTION
# ============================================================
def predict_number(image_path, model=None, verbose=True):
    """
    Predicts single or multi-digit handwritten number from an image.
    Returns:
        predicted_str: String representation (preserving leading zeros, e.g. '019724')
        avg_confidence: Average confidence percentage
        breakdown: List of tuples (digit_str, confidence_pct, (x, y, w, h))
    """
    if model is None:
        model = load_digit_model()

    binary, boxes = detect_and_segment_digits(image_path)

    breakdown = []
    for idx, (x, y, w, h) in enumerate(boxes):
        digit_input = preprocess_single_digit_crop(binary, x, y, w, h)
        batch_input = np.expand_dims(digit_input, axis=0)

        preds = model.predict(batch_input, verbose=0)[0]
        digit = str(np.argmax(preds))
        conf = float(np.max(preds) * 100.0)

        breakdown.append((digit, conf, (x, y, w, h)))

    predicted_str = "".join([d[0] for d in breakdown])
    avg_confidence = float(np.mean([d[1] for d in breakdown])) if breakdown else 0.0

    if verbose:
        print("=" * 50)
        print(f"File:               {image_path}")
        print(f"Predicted Number:   {predicted_str}")
        print(f"Average Confidence: {avg_confidence:.2f}%")
        print(f"Digits Detected:    {len(breakdown)}")
        for i, (d, c, bbox) in enumerate(breakdown, 1):
            print(f"  Digit {i}: '{d}' ({c:.2f}%) at bbox (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")
        print("=" * 50)

    return predicted_str, avg_confidence, breakdown


# ============================================================
# 4. MAIN CLI RUNNER
# ============================================================
if __name__ == "__main__":
    model = load_digit_model()

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("\n--- Evaluating all test images ---")
        image_files = sorted(glob.glob("test_images/*.jpg"))
        for img_path in image_files:
            predict_number(img_path, model=model, verbose=True)
    elif len(sys.argv) > 1:
        target_path = sys.argv[1]
        predict_number(target_path, model=model, verbose=True)
    else:
        # Default test on 7.jpg (or another sample)
        default_img = "test_images/7.jpg"
        if os.path.exists(default_img):
            predict_number(default_img, model=model, verbose=True)
        else:
            print("No target specified. Use: python predict.py <image_path> or python predict.py --all")