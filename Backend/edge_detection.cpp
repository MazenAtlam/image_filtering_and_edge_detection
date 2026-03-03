#include "binding_utils.h"
#include "intensity_data_info.h"

// Detect Edge using Canny mask
    static cv::Mat detectEdgesCanny(const cv::Mat& image, double threshold1 = 100, double threshold2 = 200) {
        cv::Mat gray = IntensityDataInfo::convertToGrayscale(image);
        cv::Mat edges;

        // Apply Canny Edge Detection
        cv::Canny(gray, edges, threshold1, threshold2);

        // Convert back to BGR to match original Python return signature
        cv::Mat result;
        cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);
        return result;
    }

    // Helper for 3x3 convolution
    static cv::Mat apply3x3EdgeFilter(const cv::Mat& image, const int kx[3][3], const int ky[3][3]) {
        cv::Mat gray = IntensityDataInfo::convertToGrayscale(image);

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

    // Detect Edge using Sobel masks dynamically generated
    static cv::Mat detectEdgesSobel(const cv::Mat& image, int ksize = 3) {
        cv::Mat gray = IntensityDataInfo::convertToGrayscale(image);

        int grid_size = (ksize == 1) ? 3 : ksize;
        std::vector<int> smooth;
        std::vector<int> deriv;

        if (ksize == 1) {
            smooth = {0, 1, 0};
            deriv = {-1, 0, 1};
        } else {
            smooth.assign(grid_size, 0);
            smooth[0] = 1;
            for (int i = 1; i < grid_size; ++i) {
                smooth[i] = 1;
                for (int j = i - 1; j > 0; --j) {
                    smooth[j] = smooth[j] + smooth[j - 1];
                }
            }

            std::vector<int> base_smooth(grid_size - 2, 0);
            base_smooth[0] = 1;
            for (int i = 1; i < grid_size - 2; ++i) {
                base_smooth[i] = 1;
                for (int j = i - 1; j > 0; --j) {
                    base_smooth[j] = base_smooth[j] + base_smooth[j - 1];
                }
            }

            deriv.assign(grid_size, 0);
            for (int i = 0; i < grid_size; ++i) {
                if (i < base_smooth.size()) deriv[i] -= base_smooth[i];
                if (i >= 2) deriv[i] += base_smooth[i - 2];
            }
        }

        std::vector<std::vector<int>> Kx(grid_size, std::vector<int>(grid_size, 0));
        std::vector<std::vector<int>> Ky(grid_size, std::vector<int>(grid_size, 0));

        double sum_pos = 0;
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                Kx[y][x] = smooth[y] * deriv[x];
                Ky[y][x] = deriv[y] * smooth[x];
                if (Kx[y][x] > 0) sum_pos += Kx[y][x];
            }
        }

        double scale = (sum_pos > 0) ? (4.0 / sum_pos) : 1.0;

        cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat padded;
        int pad = grid_size / 2;
        cv::copyMakeBorder(gray, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);

        for (int y = 0; y < gray.rows; ++y) {
            uchar* res_ptr = result.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                double px = 0.0, py = 0.0;
                for (int i = 0; i < grid_size; ++i) {
                    const uchar* pad_ptr = padded.ptr<uchar>(y + i);
                    for (int j = 0; j < grid_size; ++j) {
                        int val = pad_ptr[x + j];
                        px += val * Kx[i][j];
                        py += val * Ky[i][j];
                    }
                }
                px *= scale;
                py *= scale;
                double mag = std::sqrt(px * px + py * py);
                res_ptr[x] = cv::saturate_cast<uchar>(mag);
            }
        }
        
        cv::Mat result_bgr;
        cv::cvtColor(result, result_bgr, cv::COLOR_GRAY2BGR);
        return result_bgr;
    }

    // Detect Edge using Prewitt masks
    static cv::Mat detectEdgesPrewitt(const cv::Mat& image) {
        int kx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
        int ky[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};
        return apply3x3EdgeFilter(image, kx, ky);
    }

    // Detect Edge using Roberts cross masks
    static cv::Mat detectEdgesRoberts(const cv::Mat& image) {
        cv::Mat gray = IntensityDataInfo::convertToGrayscale(image);

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

py::array_t<unsigned char> sobel_wrapper(py::array_t<unsigned char> img, int ksize = 3) {
    auto mat = numpy_to_mat(img);
    auto res = detectEdgesSobel(mat, ksize);
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

#ifndef MAIN_BIND
PYBIND11_MODULE(edge_backend, m) {
    m.doc() = "Edge detection C++ backend";
    m.def("canny", &canny_wrapper, "Apply Canny edge detection");
    m.def("sobel", &sobel_wrapper, "Apply Sobel edge detection", py::arg("img"), py::arg("ksize") = 3);
    m.def("prewitt", &prewitt_wrapper, "Apply Prewitt edge detection");
    m.def("roberts", &roberts_wrapper, "Apply Roberts edge detection");
}
#endif
