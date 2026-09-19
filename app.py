import os
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import tensorflow as tf

# ============================================================
# CONFIG & THEME
# ============================================================

MODEL_PATH = "model/digit_model.keras"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading convolutional neural network...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

selected_file = None
preview_image = None


# ============================================================
# IMAGE PROCESSING & SEGMENTATION PIPELINE
# ============================================================

def preprocess_single_digit_crop(binary_image, x, y, w, h):
    """
    Converts a cropped digit bounding box to 28x28 MNIST format:
    1. Crop digit from binary image
    2. Place inside square canvas with centered offset
    3. Resize to 20x20
    4. Center in 28x28 with 4-pixel padding
    5. Normalize to 0.0 - 1.0 with shape (28, 28, 1)
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
    Extracts binary image and sorted digit bounding boxes.
    Retains fallback to single largest contour if only one digit is present.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open the selected image.")

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Threshold (inverse binary: dark strokes -> white foreground)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No handwritten digits detected.")

    # 4. Filter contours by area relative to the largest contour
    max_area = max(cv2.contourArea(c) for c in contours)
    min_area = max(500.0, max_area * 0.03)

    raw_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise dots
            if w >= 15 and h >= 25:
                raw_boxes.append((x, y, w, h))

    # Single-digit fallback: if no contour passed threshold, use largest
    if not raw_boxes:
        largest_contour = max(contours, key=cv2.contourArea)
        raw_boxes = [cv2.boundingRect(largest_contour)]

    # 5. Sort boxes left-to-right
    raw_boxes.sort(key=lambda b: b[0])

    # 6. Merge horizontally overlapping boxes belonging to same digit
    merged_boxes = []
    for box in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            curr_x, curr_y, curr_w, curr_h = box
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
# SELECT IMAGE
# ============================================================

def select_image():
    global selected_file
    global preview_image

    file_path = filedialog.askopenfilename(
        title="Select Handwritten Number",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )

    if not file_path:
        return

    selected_file = file_path

    try:
        # Open and resize image for preview
        image = Image.open(file_path).convert("RGB")
        image.thumbnail((700, 400), Image.Resampling.LANCZOS)

        preview_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size
        )

        image_label.configure(
            image=preview_image,
            text=""
        )

        selected_label.configure(
            text=f"Selected: {os.path.basename(file_path)}"
        )

        # Reset results
        result_label.configure(
            text="—",
            font=ctk.CTkFont(size=70, weight="bold"),
            text_color="white"
        )
        confidence_label.configure(text="Confidence: —")
        breakdown_card.pack_forget()
        breakdown_text.configure(text="")

        status_label.configure(
            text="Image loaded successfully. Click 'Predict Number' to classify.",
            text_color="#00E676"
        )

    except Exception as error:
        status_label.configure(
            text=f"Error loading image: {error}",
            text_color="#FF5555"
        )


# ============================================================
# PREDICT NUMBER (SINGLE & MULTI-DIGIT)
# ============================================================

def predict_number():
    if selected_file is None:
        status_label.configure(
            text="Please select an image first.",
            text_color="#FF5555"
        )
        return

    try:
        status_label.configure(
            text="Processing and classifying...",
            text_color="#FFAA00"
        )
        app.update()

        binary, boxes = detect_and_segment_digits(selected_file)

        breakdown = []
        for x, y, w, h in boxes:
            digit_input = preprocess_single_digit_crop(binary, x, y, w, h)
            batch_input = np.expand_dims(digit_input, axis=0)

            preds = model.predict(batch_input, verbose=0)[0]
            digit_str = str(np.argmax(preds))
            conf_pct = float(np.max(preds) * 100.0)

            breakdown.append((digit_str, conf_pct))

        # Preserve leading zeros by concatenating strings directly
        predicted_number = "".join([d[0] for d in breakdown])
        avg_confidence = float(np.mean([d[1] for d in breakdown])) if breakdown else 0.0

        # Adjust display font size for number length
        num_len = len(predicted_number)
        if num_len <= 3:
            font_size = 70
        elif num_len <= 6:
            font_size = 52
        elif num_len <= 10:
            font_size = 38
        else:
            font_size = 28

        result_label.configure(
            text=predicted_number,
            font=ctk.CTkFont(size=font_size, weight="bold"),
            text_color="#00E676"
        )

        if len(breakdown) > 1:
            confidence_label.configure(
                text=f"Average Confidence: {avg_confidence:.2f}%"
            )
            # Build breakdown display
            breakdown_lines = []
            for i, (d, c) in enumerate(breakdown, 1):
                breakdown_lines.append(f"Digit {i}:  {d}  ({c:.2f}%)")
            breakdown_text.configure(text="\n".join(breakdown_lines))
            breakdown_card.pack(fill="x", padx=25, pady=(0, 20))
        else:
            confidence_label.configure(
                text=f"Confidence: {avg_confidence:.2f}%"
            )
            breakdown_card.pack_forget()
            breakdown_text.configure(text="")

        status_label.configure(
            text="Prediction completed successfully.",
            text_color="#00E676"
        )

    except Exception as error:
        result_label.configure(
            text="!",
            font=ctk.CTkFont(size=70, weight="bold"),
            text_color="#FF5555"
        )
        confidence_label.configure(text="Prediction failed.")
        breakdown_card.pack_forget()
        breakdown_text.configure(text="")
        status_label.configure(
            text=f"Error: {error}",
            text_color="#FF5555"
        )


# ============================================================
# CLEAR
# ============================================================

def clear_image():
    global selected_file, preview_image, image_label

    selected_file = None

    # Destroy the old label completely to prevent Tkinter image reference leaks
    image_label.destroy()

    # Create a fresh CTkLabel instance
    image_label = ctk.CTkLabel(
        preview_frame,
        text="No Image Selected",
        font=ctk.CTkFont(size=18)
    )

    image_label.place(
        relx=0.5,
        rely=0.5,
        anchor="center"
    )

    preview_image = None

    selected_label.configure(
        text="Select a handwritten number image"
    )

    result_label.configure(
        text="—",
        font=ctk.CTkFont(size=70, weight="bold"),
        text_color="white"
    )

    confidence_label.configure(
        text="Confidence: —"
    )

    breakdown_card.pack_forget()
    breakdown_text.configure(text="")

    status_label.configure(
        text="Ready.",
        text_color="white"
    )


# ============================================================
# MAIN WINDOW
# ============================================================

app = ctk.CTk()
app.title("Handwritten Number Recognition")
app.geometry("1100x850")
app.minsize(900, 650)


# ============================================================
# HEADER
# ============================================================

header = ctk.CTkFrame(
    app,
    corner_radius=0,
    height=105
)
header.pack(fill="x")
header.pack_propagate(False)

title_label = ctk.CTkLabel(
    header,
    text="Handwritten Number Recognition",
    font=ctk.CTkFont(size=30, weight="bold")
)
title_label.pack(pady=(20, 3))

subtitle_label = ctk.CTkLabel(
    header,
    text="Convolutional Neural Network Based Number Classification",
    font=ctk.CTkFont(size=15)
)
subtitle_label.pack()


# ============================================================
# SCROLLABLE CONTENT
# ============================================================

scroll_frame = ctk.CTkScrollableFrame(
    app,
    corner_radius=0
)
scroll_frame.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


# ============================================================
# 1. IMAGE SECTION
# ============================================================

image_section = ctk.CTkFrame(
    scroll_frame,
    corner_radius=15
)
image_section.pack(
    fill="x",
    padx=10,
    pady=10
)

section_title = ctk.CTkLabel(
    image_section,
    text="1. Select Handwritten Number",
    font=ctk.CTkFont(size=21, weight="bold")
)
section_title.pack(
    anchor="w",
    padx=25,
    pady=(20, 10)
)

# Preview box
preview_frame = ctk.CTkFrame(
    image_section,
    height=400,
    corner_radius=12
)
preview_frame.pack(
    fill="x",
    padx=25,
    pady=10
)
preview_frame.pack_propagate(False)

image_label = ctk.CTkLabel(
    preview_frame,
    text="No Image Selected",
    font=ctk.CTkFont(size=18)
)
image_label.place(
    relx=0.5,
    rely=0.5,
    anchor="center"
)

# Selected filename
selected_label = ctk.CTkLabel(
    image_section,
    text="Select a handwritten number image",
    font=ctk.CTkFont(size=14)
)
selected_label.pack(pady=(0, 10))

# Buttons
button_frame = ctk.CTkFrame(
    image_section,
    fg_color="transparent"
)
button_frame.pack(pady=(5, 15))

select_button = ctk.CTkButton(
    button_frame,
    text="📁  Select Image",
    width=220,
    height=48,
    font=ctk.CTkFont(size=15, weight="bold"),
    command=select_image
)
select_button.grid(row=0, column=0, padx=8)

predict_button = ctk.CTkButton(
    button_frame,
    text="🤖  Predict Number",
    width=220,
    height=48,
    font=ctk.CTkFont(size=15, weight="bold"),
    command=predict_number
)
predict_button.grid(row=0, column=1, padx=8)

clear_button = ctk.CTkButton(
    button_frame,
    text="↻  Clear",
    width=140,
    height=48,
    command=clear_image
)
clear_button.grid(row=0, column=2, padx=8)

# Status
status_label = ctk.CTkLabel(
    image_section,
    text="Ready.",
    font=ctk.CTkFont(size=14)
)
status_label.pack(pady=(0, 20))


# ============================================================
# 2. RESULT SECTION
# ============================================================

result_section = ctk.CTkFrame(
    scroll_frame,
    corner_radius=15
)
result_section.pack(
    fill="x",
    padx=10,
    pady=10
)

result_title = ctk.CTkLabel(
    result_section,
    text="2. Neural Network Result",
    font=ctk.CTkFont(size=21, weight="bold")
)
result_title.pack(
    anchor="w",
    padx=25,
    pady=(20, 15)
)

result_card = ctk.CTkFrame(
    result_section,
    corner_radius=15
)
result_card.pack(
    fill="x",
    padx=25,
    pady=(0, 15)
)

result_label = ctk.CTkLabel(
    result_card,
    text="—",
    font=ctk.CTkFont(size=70, weight="bold")
)
result_label.pack(pady=(20, 0))

prediction_text = ctk.CTkLabel(
    result_card,
    text="Predicted Number",
    font=ctk.CTkFont(size=17)
)
prediction_text.pack(pady=(5, 0))

confidence_label = ctk.CTkLabel(
    result_card,
    text="Confidence: —",
    font=ctk.CTkFont(size=20, weight="bold")
)
confidence_label.pack(pady=(10, 20))

# Breakdown Card (shown when multiple digits are detected)
breakdown_card = ctk.CTkFrame(
    result_section,
    corner_radius=12,
    fg_color="#1E232A"
)

breakdown_title = ctk.CTkLabel(
    breakdown_card,
    text="Digit Breakdown",
    font=ctk.CTkFont(size=16, weight="bold")
)
breakdown_title.pack(pady=(12, 6))

breakdown_text = ctk.CTkLabel(
    breakdown_card,
    text="",
    font=ctk.CTkFont(size=15),
    justify="center"
)
breakdown_text.pack(pady=(0, 15))


# ============================================================
# 3. SYSTEM WORKFLOW SECTION
# ============================================================

workflow_section = ctk.CTkFrame(
    scroll_frame,
    corner_radius=15
)
workflow_section.pack(
    fill="x",
    padx=10,
    pady=10
)

workflow_title = ctk.CTkLabel(
    workflow_section,
    text="3. System Workflow",
    font=ctk.CTkFont(size=21, weight="bold")
)
workflow_title.pack(
    anchor="w",
    padx=25,
    pady=(20, 10)
)

workflow_text = ctk.CTkLabel(
    workflow_section,
    text=(
        "Image Selection\n"
        "↓\n"
        "OpenCV Preprocessing\n"
        "↓\n"
        "Digit Detection\n"
        "↓\n"
        "Left-to-Right Sorting\n"
        "↓\n"
        "Individual Digit Extraction\n"
        "↓\n"
        "28 × 28 MNIST Conversion\n"
        "↓\n"
        "CNN Classification\n"
        "↓\n"
        "Combine Predictions\n"
        "↓\n"
        "Predicted Number + Confidence"
    ),
    font=ctk.CTkFont(size=16),
    justify="center"
)
workflow_text.pack(pady=(5, 30))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    app.mainloop()