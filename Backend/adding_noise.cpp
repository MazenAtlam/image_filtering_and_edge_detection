#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_utils.h"
#include <string>
#include <random>

namespace py = pybind11;

class NoiseGenerator {
public:
    static cv::Mat applyUniformNoise(const cv::Mat& image, double intensity_pct) {
        // Map 0-100% to a reasonable range for uniform noise (e.g., 0 to 255)
        double range = intensity_pct * 2.55; 
        
        cv::Mat result = image.clone();
        cv::Mat random_vals = cv::Mat(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
        
        // Generate uniform noise between -range and +range
        cv::randu(random_vals, -range, range);
        
        // Add noise additively using 32-bit float to prevent overflow wrapping before saturation
        cv::Mat image_float;
        image.convertTo(image_float, CV_32F);
        image_float += random_vals;
        
        // Convert back to 8-bit unsigned char with saturation
        image_float.convertTo(result, image.type());
        return result;
    }

    static cv::Mat applyGaussianNoise(const cv::Mat& image, double intensity_pct) {
        // Standard deviation mapping
        double stddev = intensity_pct * 2.55 / 2.0; 
        double mean = 0.0;
        
        cv::Mat noise = cv::Mat(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
        cv::randn(noise, mean, stddev);
        
        cv::Mat image_float;
        image.convertTo(image_float, CV_32F);
        image_float += noise;
        
        cv::Mat result;
        image_float.convertTo(result, image.type());
        return result;
    }

    static cv::Mat applySaltAndPepperNoise(const cv::Mat& image, double intensity_pct) {
        cv::Mat result = image.clone();
        
        // Probability of a pixel being salt OR pepper
        double prob = intensity_pct / 100.0;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        int channels = image.channels();
        
        for (int y = 0; y < result.rows; ++y) {
            uchar* ptr = result.ptr<uchar>(y);
            for (int x = 0; x < result.cols; ++x) {
                double rand_val = dist(gen);
                
                // We split the probability equally between salt and pepper
                if (rand_val < prob / 2.0) {
                    // Pepper (0)
                    for (int c = 0; c < channels; ++c) {
                        ptr[x * channels + c] = 0;
                    }
                } else if (rand_val < prob) {
                    // Salt (255)
                    for (int c = 0; c < channels; ++c) {
                        ptr[x * channels + c] = 255;
                    }
                }
            }
        }
        return result;
    }
};

py::array_t<unsigned char> add_noise_wrapper(py::array_t<unsigned char> img, const std::string& noise_type, double intensity) {
    auto mat = numpy_to_mat(img);
    cv::Mat res;
    
    // Dispatch based on noise type parameter
    if (noise_type == "Uniform" || noise_type == "Uniform Noise") {
        res = NoiseGenerator::applyUniformNoise(mat, intensity);
    } else if (noise_type == "Gaussian" || noise_type == "Gaussian Noise") {
        res = NoiseGenerator::applyGaussianNoise(mat, intensity);
    } else if (noise_type == "Salt & Pepper" || noise_type == "Salt and Pepper") {
        res = NoiseGenerator::applySaltAndPepperNoise(mat, intensity);
    } else {
        // Fallback to original if invalid type
        res = mat.clone(); 
    }
    
    return mat_to_numpy(res);
}

#ifndef MAIN_BIND
PYBIND11_MODULE(noise_backend, m) {
    m.doc() = "Noise generation C++ backend";
    m.def("add_noise", &add_noise_wrapper, "Add noise to an image dynamically based on type and intensity",
          py::arg("image"), py::arg("noise_type"), py::arg("intensity"));
}
#endif
