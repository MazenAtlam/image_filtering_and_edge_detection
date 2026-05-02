# Image Filtering and Edge Detection

A desktop application that provides a comprehensive suite of image processing tools. The project utilizes a high-performance **C++ backend** to handle computationally intensive image processing algorithms, seamlessly integrated with an intuitive **Python (PyQt) frontend**.

## Features

Based on the core backend implementation, the application supports:

- **Noise Injection & Filtering:** Add artificial noise (e.g., Gaussian, Salt & Pepper) and apply smoothing filters to clean noisy images.
- **Edge Detection:** Detect boundaries and sharp edges in images using standard gradient operators.
- **Image Enhancement:** Improve visual quality and adjust contrast for low-contrast images.
- **Frequency Domain Filters:** Apply low-pass and high-pass filters using frequency domain transformations.
- **Hybrid Images:** Generate hybrid images by combining the low frequencies of one image with the high frequencies of another.
- **Intensity Data Analysis:** Extract and analyze image intensity histograms and metrics.

## Prerequisites

To build and run this project, you will need the following tools installed on your system:

### Backend (C++)

- **C++ Compiler:** GCC, Clang, or MSVC (C++17 or higher recommended).
- **CMake:** Version 3.10 or higher for building the backend.
- **OpenCV C++:** OpenCV might be needed for image matrix operations in C++.

### Frontend (Python)

- **Python:** Version 3.8 or higher.
- **Python Packages:**
  - `PyQt5` or `PyQt6` (for the user interface)
  - `numpy`
  - `opencv-python` (cv2)

## Quick Start

Follow these steps to get the application up and running on your local machine:

1. **Clone the repository**

    ```bash
    git clone [https://github.com/mazenatlam/image_filtering_and_edge_detection.git](https://github.com/mazenatlam/image_filtering_and_edge_detection.git)
    cd image_filtering_and_edge_detection
    ```

2. **Build the C++ Backend**
Navigate to the `Backend` directory and use CMake to build the project files:

    ```bash
    cd Backend
    mkdir build
    cd build
    cmake ..
    make  # On Windows, you might use 'cmake --build .'
    ```

3. **Install Python Dependencies**
Ensure your Python environment is set up and install the required UI/processing libraries:

    ```bash
    pip install PyQt5 numpy opencv-python
    ```

4. **Run the Application**
Navigate to the `Frontend` directory and launch the main Python script:

    ```bash
    cd ../../Frontend
    python front.py
    ```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
