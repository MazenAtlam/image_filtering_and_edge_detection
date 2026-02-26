#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_utils.h"
#include <string>

namespace py = pybind11;

class SpatialFilter {
public:
    static cv::Mat applyAverageFilter(const cv::Mat& image, int kernel_size) {
        cv::Mat result;
        // OpenCV blur acts as a normalized box filter
        cv::blur(image, result, cv::Size(kernel_size, kernel_size));
        return result;
    }

    static cv::Mat applyGaussianFilter(const cv::Mat& image, int kernel_size) {
        cv::Mat result;
        // Gaussian blur. Setting sigmaX and sigmaY to 0 lets OpenCV calculate it automatically from the kernel size
        cv::GaussianBlur(image, result, cv::Size(kernel_size, kernel_size), 0, 0);
        return result;
    }

    static cv::Mat applyMedianFilter(const cv::Mat& image, int kernel_size) {
        cv::Mat result;
        // Median filter is non-linear and replaces each pixel with the median of its neighbors
        cv::medianBlur(image, result, kernel_size);
        return result;
    }
};

py::array_t<unsigned char> apply_filter_wrapper(py::array_t<unsigned char> img, const std::string& filter_type, int kernel_size) {
    auto mat = numpy_to_mat(img);
    cv::Mat res;
    
    // Dispatch based on filter type parameter
    if (filter_type == "Average Filter" || filter_type == "Average") {
        res = SpatialFilter::applyAverageFilter(mat, kernel_size);
    } else if (filter_type == "Gaussian Filter" || filter_type == "Gaussian") {
        res = SpatialFilter::applyGaussianFilter(mat, kernel_size);
    } else if (filter_type == "Median Filter" || filter_type == "Median") {
        res = SpatialFilter::applyMedianFilter(mat, kernel_size);
    } else {
        // Fallback
        res = mat.clone(); 
    }
    
    return mat_to_numpy(res);
}

PYBIND11_MODULE(filter_backend, m) {
    m.doc() = "Spatial Domain filtering C++ backend";
    m.def("apply_filter", &apply_filter_wrapper, "Apply spatial filters based on type and kernel size",
          py::arg("image"), py::arg("filter_type"), py::arg("kernel_size"));
}
