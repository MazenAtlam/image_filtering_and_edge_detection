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

    // General Helper for edge filter convolutions using cv::Mat kernels (Assumes odd symmetric kernels)
    static cv::Mat applyEdgeFilter(const cv::Mat& image, const cv::Mat& Kx, const cv::Mat& Ky, double scale = 1.0) {
        cv::Mat gray = IntensityDataInfo::convertToGrayscale(image);

        cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
        cv::Mat padded;

        // Assumes Kx and Ky are odd and square
        int ksize = Kx.rows;
        int pad = ksize / 2;

        std::vector<std::vector<int>> kx_vec(ksize, std::vector<int>(ksize));
        std::vector<std::vector<int>> ky_vec(ksize, std::vector<int>(ksize));
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                kx_vec[i][j] = Kx.at<int>(i, j);
                ky_vec[i][j] = Ky.at<int>(i, j);
            }
        }

        cv::copyMakeBorder(gray, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);

        for (int y = 0; y < gray.rows; ++y) {
            uchar* res_ptr = result.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                double px = 0.0, py = 0.0;
                for (int i = 0; i < ksize; ++i) {
                    const uchar* pad_ptr = padded.ptr<uchar>(y + i);
                    for (int j = 0; j < ksize; ++j) {
                        int val = pad_ptr[x + j];
                        px += val * kx_vec[i][j];
                        py += val * ky_vec[i][j];
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

        cv::Mat Kx_mat(grid_size, grid_size, CV_32S);
        cv::Mat Ky_mat(grid_size, grid_size, CV_32S);

        double sum_pos = 0;
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                int vx = smooth[y] * deriv[x];
                int vy = deriv[y] * smooth[x];
                Kx_mat.at<int>(y, x) = vx;
                Ky_mat.at<int>(y, x) = vy;
                if (vx > 0) sum_pos += vx;
            }
        }

        double scale = (sum_pos > 0) ? (4.0 / sum_pos) : 1.0;
        return applyEdgeFilter(image, Kx_mat, Ky_mat, scale);
    }

    // Detect Edge using Prewitt masks
    static cv::Mat detectEdgesPrewitt(const cv::Mat& image) {
        cv::Mat kx = (cv::Mat_<int>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
        cv::Mat ky = (cv::Mat_<int>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
        return applyEdgeFilter(image, kx, ky);
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
