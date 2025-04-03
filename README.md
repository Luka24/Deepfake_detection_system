# Deepfake Detection System

An automated system for detection and analysis of deepfake images and videos, based on advanced machine learning and computer vision techniques.

## Description

This project implements a comprehensive system for recognizing deepfake content using deep neural networks. The system is capable of detecting visual anomalies and artifacts characteristic of computer-generated or manipulated faces in images and videos.

## Features

- **Image Deepfake Detection**: Analysis of individual images and faces
- **Video Deepfake Detection**: Analysis of frame sequences for improved accuracy
- **Custom Training**: Ability to train the model on your own data
- **Facial Feature Analysis**: Verification of facial expression naturalness and movements
- **Eye Blinking Analysis**: Detection of unnatural blinking patterns common in deepfakes
- **Result Visualization**: Display of detection results with annotations on images and videos
- **Model Evaluation**: Tools for measuring model accuracy

## Requirements

The program requires:

- Python 3.7 or newer
- TensorFlow 2.x
- OpenCV 4.x
- dlib
- numpy
- Matplotlib
- pandas
- scikit-learn
- tqdm

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/username/deepfake-detection.git
cd deepfake-detection
```

### 2. Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Download the Dlib facial landmark predictor:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

### Detecting Deepfakes in Images
```python
from deepfake_detector import DeepfakeDetector

# Initialize the detector
detector = DeepfakeDetector(model_path="model/deepfake_model.h5")

# Detect on a single image
result = detector.detect_from_image("path/to/image.jpg")

if result:
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Save the annotated image
    import cv2
    cv2.imwrite("result.jpg", result["visualization"])
```

### Detecting Deepfakes in Videos
```python
# Detect on a video file
result = detector.detect_from_video(
    "path/to/video.mp4", 
    output_path="video_result.mp4"
)

if result:
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

### Training the Model
```python
# Train on your own data
history = detector.train_model(
    real_dir="dataset/real_images",
    fake_dir="dataset/fake_images",
    epochs=50,
    batch_size=32
)

# Evaluate the model
eval_results = detector.evaluate_model(
    test_dir_real="test_data/real",
    test_dir_fake="test_data/fake"
)
```

## Code Structure

- **DeepfakeDetector**: Main detection class
- **detect_faces()**: Face detection in images
- **extract_facial_features()**: Facial feature extraction
- **detect_from_image()**: Deepfake detection in images
- **detect_from_video()**: Deepfake detection in videos
- **train_model()**: Model training
- **evaluate_model()**: Model evaluation
- **analyze_artifacts()**: Visual artifact analysis
- **analyze_eye_blinking()**: Eye blinking pattern analysis

## Result Examples
The system returns results as a dictionary containing:

- **prediction**: "REAL" or "FAKE"
- **confidence**: Confidence value (0-1)
- **visualization**: Image with result annotations
- Additional metrics and values for more detailed analysis

## Notes

- For best results, we recommend using a GPU for training and faster detection.
- System accuracy depends on the quality of training data.
- The system is intended for research and educational purposes.


