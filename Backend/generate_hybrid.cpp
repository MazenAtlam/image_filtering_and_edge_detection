#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_utils.h"

#define FREQ_DONT_BIND
#include "frequency_filters.cpp"

namespace py = pybind11;

class HybridGenerator {
public:
    // Make the Hybrid Image
    static cv::Mat createHybridImage(const cv::Mat& img_a, const cv::Mat& img_b, int radius_a, int radius_b) {
        cv::Mat b_resized;
        
        // Ensure images are the same size
        if (img_a.size() != img_b.size()) {
            cv::resize(img_b, b_resized, img_a.size());
        } else {
            b_resized = img_b.clone();
        }

        // Apply Low-pass to Image A and High-pass to Image B
        cv::Mat low_pass_a = FrequencyFilters::applyFFTFilter(img_a, "low_pass", radius_a);
        cv::Mat high_pass_b = FrequencyFilters::applyFFTFilter(b_resized, "high_pass", radius_b);

        // Convert to float to avoid overflow/underflow during subtraction and addition
        cv::Mat low_float, high_float;
        low_pass_a.convertTo(low_float, CV_32F);
        high_pass_b.convertTo(high_float, CV_32F);

        // 'applyFFTFilter' normalizes its output to [0, 255], which gives the high-pass image
        // an artificial DC offset (mean around ~128). We must subtract this mean so that
        // the high-pass details are centered around 0 before adding to the low-pass image.
        cv::Scalar mean_val = cv::mean(high_float);
        high_float -= mean_val;

        // Combine them
        cv::Mat hybrid_float = low_float + high_float;
        
        cv::Mat hybrid;
        hybrid_float.convertTo(hybrid, CV_8U); // This automatically saturates above 255 and below 0
        
        return hybrid;
    }
};

py::array_t<unsigned char> create_hybrid_wrapper(py::array_t<unsigned char> img_a, py::array_t<unsigned char> img_b, int radius_a, int radius_b) {
    auto mat_a = numpy_to_mat(img_a);
    auto mat_b = numpy_to_mat(img_b);
    auto res = HybridGenerator::createHybridImage(mat_a, mat_b, radius_a, radius_b);
    return mat_to_numpy(res);
}

PYBIND11_MODULE(hybrid_backend, m) {
    m.doc() = "Hybrid Image generation C++ backend";
    m.def("create_hybrid", &create_hybrid_wrapper, "Create hybrid image from two inputs",
          py::arg("img_a"), py::arg("img_b"), py::arg("radius_a"), py::arg("radius_b"));
}
