#pragma once
#include <opencv2/opencv.hpp>

class IntensityDataInfo {
public:
    // 1. Grayscale Conversion
    static cv::Mat convertToGrayscale(const cv::Mat& image) {
        if (image.channels() == 3) {
            cv::Mat gray;
            // Note: Depending on frontend (RGB vs BGR representation in NumPy), we might need COLOR_RGB2GRAY
            // Assuming default OpenCV BGR order for CV_8UC3 internally. If rgb, result is visually identical for grayscale.
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return image.clone(); // Already grayscale
    }
};
