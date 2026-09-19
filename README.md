# Handwritten Number Recognition Using Convolutional Neural Networks

A complete end-to-end Machine Learning project that classifies single-digit and multi-digit handwritten numbers from real-world phone photographs using OpenCV and a deep Convolutional Neural Network (CNN) trained on MNIST.

---

## 1. Technologies & Libraries Used

| Technology | Version / Purpose |
| :--- | :--- |
| **Python** | 3.10 / 3.11 (Core language) |
| **TensorFlow / Keras** | 2.16+ (Deep learning framework: CNN training, inference, data augmentation) |
| **OpenCV (`opencv-python`)** | 4.9+ (Image preprocessing: grayscale, thresholding, contour detection, bounding boxes) |
| **CustomTkinter** | 6.0+ (Modern desktop GUI with dark theme and responsive layout) |
| **Pillow (`PIL`)** | 10.0+ (Image file loading and preview rendering) |
| **NumPy** | 1.26+ (Matrix manipulation, array reshaping, normalization) |
| **Matplotlib** | 3.8+ (Visualizations and debugging plots) |

---

## 2. Project Directory Structure

```text
HandwrittenDigitRecognition/
│
├── model/
│   └── digit_model.keras       # Trained CNN model weights
├── test_images/                # Real phone photos of handwritten digits (0-8, multi-digits)
├── requirements.txt            # Python package dependencies
├── train_model.py              # CNN architecture, data augmentation, & MNIST training script
├── predict.py                  # CLI testing script (single & multi-digit evaluation)
├── app.py                      # CustomTkinter GUI Desktop Application
└── README.md                   # Complete project documentation
```

---

## 3. How the System Works

### A. Neural Network Architecture (`train_model.py`)
The model is a deep Convolutional Neural Network (CNN) built with Keras:

```text
Input: (28, 28, 1)
  ↓
Data Augmentation:
  - RandomRotation(±8°)
  - RandomTranslation(±8%, ±8%)
  - RandomZoom(±8%)
  ↓
Conv Block 1:
  - Conv2D (32 filters, 3x3 kernel, ReLU, same padding)
  - BatchNormalization()
  - Conv2D (32 filters, 3x3 kernel, ReLU)
  - MaxPooling2D (2x2 pool size)
  ↓
Conv Block 2:
  - Conv2D (64 filters, 3x3 kernel, ReLU, same padding)
  - BatchNormalization()
  - Conv2D (64 filters, 3x3 kernel, ReLU)
  - MaxPooling2D (2x2 pool size)
  ↓
Classifier Head:
  - Flatten()
  - Dense (128 units, ReLU)
  - Dropout (0.35)
  - Dense (10 units, Softmax)
```
- **Why Data Augmentation?** Real-world phone photos have slight camera angles, stroke thicknesses, and paper shifts. Augmentation forces the CNN to learn spatial features rather than memorizing pixel coordinates.
- **Accuracy:** Reaches **99.21% test accuracy** on MNIST and **100% accuracy** on the real test image set.

---

### B. Preprocessing & Multi-Digit Pipeline (`predict.py` & `app.py`)

1. **Grayscale Conversion**: Image is loaded via OpenCV and converted to grayscale (`cv2.cvtColor`).
2. **Inverse Binary Thresholding**:
   - `cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)`
   - Dark pen strokes become white pixels (`255`) on a black background (`0`), matching MNIST conventions.
3. **External Contour Detection**: `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)` locates digit boundaries.
4. **Noise Filtering**: Removes small artifacts, shadows, or dust dots (must be $\ge 3\%$ of largest contour area).
5. **Left-to-Right Sorting**: Sorts all bounding boxes horizontally by their `x` coordinate.
6. **Stroke Merging**: Merges bounding boxes if strokes of the same digit overlap horizontally.
7. **Single-Digit Fallback**: If only 1 contour exists, it seamlessly defaults to single-digit mode.
8. **MNIST Standardization**:
   - Crops each digit bounding box.
   - Centers inside a square canvas (preserving aspect ratio).
   - Resizes to $20 \times 20$ pixels.
   - Pads to $28 \times 28$ with a 4-pixel border.
   - Normalizes pixel values to `[0.0, 1.0]`.
9. **Inference & String Preservation**:
   - Each digit is classified by the CNN.
   - Predictions are concatenated as a **string** (`"019724"`), ensuring **leading zeros are strictly preserved**.
   - Calculates both the overall average confidence and per-digit breakdown.

---

## 4. How to Run on a New PC or Laptop

### Step 1: Install Python
- Ensure **Python 3.10** or **Python 3.11** is installed.
- Download from: [python.org](https://www.python.org/downloads/)
- **Important:** Check the box **"Add python.exe to PATH"** during installation.

---

### Step 2: Copy / Clone the Project Folder
Open PowerShell or Command Prompt, and navigate to the project directory:
```powershell
cd C:\Users\<YourUsername>\Desktop\HandwrittenDigitRecognition
```

---

### Step 3: Create a Virtual Environment
```powershell
python -m venv venv
```

---

### Step 4: Activate the Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```
- **On Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **On Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```
*(When activated, your terminal prompt will show `(venv)` at the beginning).*

---

### Step 5: Install Dependencies
Install all required libraries with one command:
```powershell
pip install -r requirements.txt
```

---

### Step 6: (Optional) Train the CNN Model
The repository already includes the trained model weights in `model/digit_model.keras`.
If you want to retrain from scratch:
```powershell
python train_model.py
```
*Takes ~1 to 2 minutes on CPU. Prints evaluation accuracy (~99.2%) and saves to `model/digit_model.keras`.*

---

### Step 7: Test Predictions via CLI
To test all sample images in `test_images/`:
```powershell
python predict.py --all
```
To test a specific image:
```powershell
python predict.py test_images/7.jpg
```

---

### Step 8: Launch the Desktop GUI Application
```powershell
python app.py
```

#### How to Use the GUI:
1. Click **"📁 Select Image"** and choose any photo from `test_images/` or any handwritten photo.
2. Click **"🤖 Predict Number"**:
   - Displays the classified number (e.g. `019724` or `7`).
   - Displays overall confidence percentage.
   - If multi-digit, displays the **Digit Breakdown** card showing each digit and its individual confidence.
3. Click **"↻ Clear"** to reset and test another image.
