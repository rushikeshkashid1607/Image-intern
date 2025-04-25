# Image Recognition System (Image-intern)

This project implements a real-time image recognition system using Python. It leverages **YOLOv5** for object detection and **DeepFace** with **dlib** for facial expression recognition, processing video feed from a webcam using **OpenCV**. The system detects objects in the scene and identifies facial expressions in real-time, displaying results on the video stream.

## Features
- **Object Detection**: Uses YOLOv5 to detect and classify objects in real-time with bounding boxes and confidence scores.
- **Facial Expression Recognition**: Detects faces using dlib and analyzes emotions using DeepFace, displaying the dominant emotion.
- **Real-Time Processing**: Processes webcam video feed for live detection and visualization.
- **Customizable**: Easily adaptable for different detection models or additional features.

## Prerequisites
- **Python**: Version 3.12.6
- **Operating System**: Windows (tested on Windows 10/11)
- **Webcam**: A working webcam connected to your system
- **Dependencies**: Listed in `requirements.txt`
- **Additional Files**: 
  - `shape_predictor_68_face_landmarks.dat` (for dlib facial landmark detection)

## Setup Instructions

Follow these steps to set up the project on a Windows system with Python 3.12.6:

1. **Prepare Your System**:
   - **Verify Python Installation**:
     - Ensure Python 3.12.6 is installed. Run in PowerShell or Command Prompt:
       ```bash
       python --version
       ```
       Expected output: `Python 3.12.6`.
     - If not installed, download from [python.org](https://www.python.org/downloads/release/python-3126/) and check "Add Python to PATH" during installation.
   - **Ensure a Webcam is Available**:
     - Confirm a working webcam is connected, as the script uses it for real-time video feed.
   - **Install Visual Studio Build Tools (Optional for dlib)**:
     - If building `dlib` from source, download Visual Studio Build Tools from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload.
     - Alternatively, use a precompiled `dlib` wheel to avoid this step.
   - **Install CMake (Optional for dlib)**:
     - If building `dlib` from source, download CMake from [cmake.org](https://cmake.org/download/) (Windows installer).
     - Select "Add CMake to the system PATH for all users" during installation.
     - Verify: `cmake --version` (should be 3.5 or higher).

2. **Create the Project Directory**:
   - Create and navigate to the project directory:
     ```bash
     mkdir "D:\Python Project\Image-intern"
     cd "D:\Python Project\Image-intern"
     ```

3. **Set Up a Virtual Environment**:
   - Create a virtual environment:
     ```bash
     python -m venv .venv
     ```
   - Activate it:
     ```bash
     .venv\Scripts\activate
     ```
     Your prompt should show `(.venv)`.

4. **Save the Main Script**:
   - Create `realtime_image_recognition.py` in the project directory.
   - Copy the script from the project source or ensure it contains the real-time detection logic (object detection with YOLOv5, facial expression recognition with DeepFace/dlib).

5. **Save the Requirements File**:
   - Create `requirements.txt` in the project directory with:
     ```
     opencv-python==4.10.0.84
     numpy==1.26.4
     torch==2.4.1
     torchvision==0.19.1
     deepface==0.0.93
     dlib==19.24.2
     ultralytics==8.3.15
     Pillow==10.4.0
     pandas==2.2.3
     tqdm==4.66.5
     requests==2.32.3
     matplotlib==3.9.2
     scipy==1.14.1
     pyyaml==6.0.2
     tensorflow==2.19.0
     tf-keras==2.19.0
     keras==3.5.0
     gdown==5.2.0
     ```
   - Save the file.

6. **Install Dependencies**:
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

7. **Install dlib (if Needed)**:
   - Install `dlib` using a precompiled wheel:
     ```bash
     pip install dlib-19.24.2-cp312-cp312-win_amd64.whl
     ```
   - Download the wheel from [PyPI](https://pypi.org/project/dlib/#files) or [christophgysin/pypi-wheels](https://github.com/christophgysin/pypi-wheels) if not already present.
   - Verify:
     ```bash
     python -c "import dlib; print(dlib.__version__)"
     ```
     Expected output: `19.24.2`.

8. **Download the Shape Predictor File**:
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib's official source](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - Extract if compressed (`.bz2`) and place in `D:\Python Project\Image-intern`.
   - Verify presence:
     ```bash
     dir shape_predictor_68_face_landmarks.dat
     ```

9. **Verify Installation**:
   - Test dependencies:
     ```bash
     python -c "import cv2, torch, deepface, dlib; print('Dependencies loaded successfully')"
     ```
     Expected output: `Dependencies loaded successfully`.

10. **Run the Script**:
    - Execute the real-time recognition script:
      ```bash
      python realtime_image_recognition.py
      ```
    - The webcam feed should display green bounding boxes for objects and red bounding boxes with emotion labels for faces.
    - Press `q` to exit.

11. **Troubleshooting**:
    - **Webcam Not Found**:
      - Ensure webcam is connected. Try `cv2.VideoCapture(1)` if `cv2.VideoCapture(0)` fails.
    - **Dependency Errors**:
      - Reinstall problematic packages:
        ```bash
        pip install tensorflow==2.19.0 tf-keras==2.19.0
        ```
    - **dlib Issues**:
      - Use the precompiled wheel or install Visual Studio Build Tools and CMake.
    - **Performance Issues**:
      - Use a CUDA-enabled GPU with:
        ```bash
        pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
        ```

## Usage
- Run the script:
  ```bash
  python realtime_image_recognition.py
  ```
- **Expected Output**:
  - Objects: Green bounding boxes with labels (e.g., "person 0.95").
  - Faces: Red bounding boxes with emotions (e.g., "happy").
- Press `q` to close the window.

## Project Structure
```
Image-intern/
├── .venv/                           # Virtual environment
├── realtime_image_recognition.py    # Main script
├── requirements.txt                 # Dependency list
├── shape_predictor_68_face_landmarks.dat  # dlib shape predictor
├── dlib-19.24.2-cp312-cp312-win_amd64.whl  # Optional, if used
└── README.md                        # Project documentation
```

## Dependencies
- **OpenCV**: Webcam handling and image processing.
- **YOLOv5 (ultralytics)**: Object detection.
- **DeepFace**: Facial expression analysis.
- **dlib**: Face detection and landmark prediction.
- **TensorFlow/tf-keras**: Backend for DeepFace models.

## Notes
- Optimized for Python 3.12.6 on Windows. Adjust for other platforms as needed.
- Ensure sufficient resources (4GB+ RAM, modern CPU) for real-time processing.
- GPU acceleration requires CUDA-compatible `torch` and `tensorflow`.

## Contributing
Fork, submit issues, or contribute via pull requests.

## License
MIT License.

## Acknowledgments
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [DeepFace](https://github.com/serengil/deepface)
- [dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)