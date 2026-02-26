#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

// Helper to convert pybind11 numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<unsigned char>& input) {
    py::buffer_info buf = input.request();
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;
    int type = channels == 3 ? CV_8UC3 : CV_8UC1;
    cv::Mat mat(buf.shape[0], buf.shape[1], type, (unsigned char*)buf.ptr);
    return mat;
}

// Helper to convert cv::Mat to pybind11 numpy array
py::array_t<unsigned char> mat_to_numpy(const cv::Mat& input) {
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

// Detect Edge using Canny mask
    static cv::Mat detectEdgesCanny(const cv::Mat& image, double threshold1 = 100, double threshold2 = 200) {
        cv::Mat gray, edges;
        
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Apply Canny Edge Detection
        cv::Canny(gray, edges, threshold1, threshold2);

        // Convert back to BGR to match original Python return signature
        cv::Mat result;
        cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
        return result;
    }

    // Helper for 3x3 convolution
    static cv::Mat apply3x3EdgeFilter(const cv::Mat& image, const int kx[3][3], const int ky[3][3]) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat padded;
        // Pad with 1 pixel on all sides
        cv::copyMakeBorder(gray, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);

        for (int y = 0; y < gray.rows; ++y) {
            uchar* res_ptr = result.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                double px = 0.0, py = 0.0;
                for (int i = 0; i < 3; ++i) {
                    const uchar* pad_ptr = padded.ptr<uchar>(y + i);
                    for (int j = 0; j < 3; ++j) {
                        int val = pad_ptr[x + j];
                        px += val * kx[i][j];
                        py += val * ky[i][j];
                    }
                }
                double mag = std::sqrt(px * px + py * py);
                res_ptr[x] = cv::saturate_cast<uchar>(mag);
            }
        }
        cv::Mat result_bgr;
        cv::cvtColor(result, result_bgr, cv::COLOR_GRAY2BGR);
        return result_bgr;
    }

    // Detect Edge using Sobel masks
    static cv::Mat detectEdgesSobel(const cv::Mat& image) {
        int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        return apply3x3EdgeFilter(image, kx, ky);
    }

    // Detect Edge using Prewitt masks
    static cv::Mat detectEdgesPrewitt(const cv::Mat& image) {
        int kx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
        int ky[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};
        return apply3x3EdgeFilter(image, kx, ky);
    }

    // Detect Edge using Roberts cross masks
    static cv::Mat detectEdgesRoberts(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat padded;
        // Pad with 1 pixel on bottom and right
        cv::copyMakeBorder(gray, padded, 0, 1, 0, 1, cv::BORDER_REPLICATE);

        int kx[2][2] = {{1, 0}, {0, -1}};
        int ky[2][2] = {{0, 1}, {-1, 0}};

        for (int y = 0; y < gray.rows; ++y) {
            uchar* res_ptr = result.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                double px = 0.0, py = 0.0;
                for (int i = 0; i < 2; ++i) {
                    const uchar* pad_ptr = padded.ptr<uchar>(y + i);
                    for (int j = 0; j < 2; ++j) {
                        int val = pad_ptr[x + j];
                        px += val * kx[i][j];
                        py += val * ky[i][j];
                    }
                }
                double mag = std::sqrt(px * px + py * py);
                res_ptr[x] = cv::saturate_cast<uchar>(mag);
            }
        }
        cv::Mat result_bgr;
        cv::cvtColor(result, result_bgr, cv::COLOR_GRAY2BGR);
        return result_bgr;
    }

// Pybind11 Wrappers
py::array_t<unsigned char> canny_wrapper(py::array_t<unsigned char> img, double t1, double t2) {
    auto mat = numpy_to_mat(img);
    auto res = detectEdgesCanny(mat, t1, t2);
    return mat_to_numpy(res);
}

py::array_t<unsigned char> sobel_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = detectEdgesSobel(mat);
    return mat_to_numpy(res);
}

py::array_t<unsigned char> prewitt_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = detectEdgesPrewitt(mat);
    return mat_to_numpy(res);
}

py::array_t<unsigned char> roberts_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = detectEdgesRoberts(mat);
    return mat_to_numpy(res);
}

PYBIND11_MODULE(edge_backend, m) {
    m.doc() = "Edge detection C++ backend";
    m.def("canny", &canny_wrapper, "Apply Canny edge detection");
    m.def("sobel", &sobel_wrapper, "Apply Sobel edge detection");
    m.def("prewitt", &prewitt_wrapper, "Apply Prewitt edge detection");
    m.def("roberts", &roberts_wrapper, "Apply Roberts edge detection");
}
