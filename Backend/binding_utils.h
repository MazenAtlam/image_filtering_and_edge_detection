#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;

// Helper to convert pybind11 numpy array to cv::Mat
inline cv::Mat numpy_to_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf = input.request();
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;
    int type = channels == 3 ? CV_8UC3 : CV_8UC1;
    cv::Mat mat(buf.shape[0], buf.shape[1], type, (unsigned char*)buf.ptr);
    return mat;
}

// Helper to convert cv::Mat to pybind11 numpy array
inline py::array_t<unsigned char> mat_to_numpy(const cv::Mat& input) {
    if (input.channels() == 3) {
        py::array_t<unsigned char> dst({ input.rows, input.cols, 3 });
        py::buffer_info buf = dst.request();
        std::memcpy(buf.ptr, input.data, input.total() * 3);
        return dst;
    } else {
        py::array_t<unsigned char> dst({ input.rows, input.cols });
        py::buffer_info buf = dst.request();
        std::memcpy(buf.ptr, input.data, input.total());
        return dst;
    }
}
