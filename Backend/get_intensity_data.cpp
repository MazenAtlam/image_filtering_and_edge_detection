#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_utils.h"
#include <vector>

namespace py = pybind11;

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

// Pybind11 Wrappers

py::array_t<unsigned char> to_grayscale_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = IntensityDataInfo::convertToGrayscale(mat);
    return mat_to_numpy(res);
}

// Calculate Histogram and return as numpy array of shape (channels, 256)
py::array_t<int> histogram_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    int channels = mat.channels();
    
    py::array_t<int> result({channels, 256});
    py::buffer_info buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    
    // Initialize to 0
    std::fill(ptr, ptr + (channels * 256), 0);
    
    if (channels == 3) {
        for (int y = 0; y < mat.rows; ++y) {
            const cv::Vec3b* src_ptr = mat.ptr<cv::Vec3b>(y);
            for (int x = 0; x < mat.cols; ++x) {
                ptr[0 * 256 + src_ptr[x][0]]++;
                ptr[1 * 256 + src_ptr[x][1]]++;
                ptr[2 * 256 + src_ptr[x][2]]++;
            }
        }
    } else {
        for (int y = 0; y < mat.rows; ++y) {
            const uchar* src_ptr = mat.ptr<uchar>(y);
            for (int x = 0; x < mat.cols; ++x) {
                ptr[src_ptr[x]]++;
            }
        }
    }
    
    return result;
}

// Calculate CDF based on the calculated histogram
py::array_t<int> cdf_wrapper(py::array_t<unsigned char> img) {
    // Generate the histogram first
    py::array_t<int> hist = histogram_wrapper(img);
    py::buffer_info hist_buf = hist.request();
    int* hist_ptr = static_cast<int*>(hist_buf.ptr);
    
    int channels = hist.shape(0);
    
    py::array_t<int> result({channels, 256});
    py::buffer_info res_buf = result.request();
    int* res_ptr = static_cast<int*>(res_buf.ptr);
    
    for (int c = 0; c < channels; ++c) {
        res_ptr[c * 256 + 0] = hist_ptr[c * 256 + 0];
        for (int i = 1; i < 256; ++i) {
            res_ptr[c * 256 + i] = res_ptr[c * 256 + i - 1] + hist_ptr[c * 256 + i];
        }
    }
    
    return result;
}

PYBIND11_MODULE(intensity_backend, m) {
    m.doc() = "Intensity data extraction C++ backend for Histogram and CDF";
    m.def("to_grayscale", &to_grayscale_wrapper, "Convert image to grayscale");
    m.def("calculate_histogram", &histogram_wrapper, "Calculate 256-bin histogram for each channel");
    m.def("calculate_cdf", &cdf_wrapper, "Calculate Cumulative Distribution Function for each channel");
}
