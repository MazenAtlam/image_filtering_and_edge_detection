#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

class ImageEnhancer {
public:
    // 1. Histogram Equalization (for Grayscale Images)
    static cv::Mat equalizeHistogram(const cv::Mat& image) {
        cv::Mat gray;
        // Convert to grayscale if it's a color image. 
        // Note: For color histogram equalization typically you'd convert to YUV/HSV and equalize the Lightness/Value channel.
        // For simplicity and standard assignment requirements, we apply it on the grayscale version.
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // 1. Calculate Histogram
        int hist[256] = {0};
        for (int y = 0; y < gray.rows; ++y) {
            const uchar* ptr = gray.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                hist[ptr[x]]++;
            }
        }

        // 2. Calculate Cumulative Distribution Function (CDF)
        int cdf[256] = {0};
        cdf[0] = hist[0];
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        // 3. Find the minimum non-zero value in the CDF
        int cdf_min = 0;
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] > 0) {
                cdf_min = cdf[i];
                break;
            }
        }

        // 4. Create the equalization mapping function (LUT - Look Up Table)
        int total_pixels = gray.rows * gray.cols;
        uchar lut[256];
        for (int i = 0; i < 256; ++i) {
            // General Formula: h(v) = round( (cdf(v) - cdf_min) / (M*N - cdf_min) * (L - 1) )
            float normalized_val = static_cast<float>(cdf[i] - cdf_min) / (total_pixels - cdf_min);
            lut[i] = cv::saturate_cast<uchar>(std::round(normalized_val * 255.0f));
        }

        // 5. Apply the mapping to create the new image
        cv::Mat result(gray.size(), gray.type());
        for (int y = 0; y < gray.rows; ++y) {
            const uchar* src_ptr = gray.ptr<uchar>(y);
            uchar* dst_ptr = result.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                dst_ptr[x] = lut[src_ptr[x]];
            }
        }

        // Convert back to BGR for consistent frontend display
        cv::Mat result_bgr;
        cv::cvtColor(result, result_bgr, cv::COLOR_GRAY2BGR);
        return result_bgr;
    }

    // 2. Image Normalization (Contrast Stretching)
    static cv::Mat normalizeImage(const cv::Mat& image) {
        cv::Mat gray;
        if (image.channels() == 3) {
             cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
             gray = image.clone();
        }

        // 1. Find the min (I_min) and max (I_max) intensity values in the current image
        uchar I_min = 255;
        uchar I_max = 0;
        
        for (int y = 0; y < gray.rows; ++y) {
            const uchar* ptr = gray.ptr<uchar>(y);
            for (int x = 0; x < gray.cols; ++x) {
                if (ptr[x] < I_min) I_min = ptr[x];
                if (ptr[x] > I_max) I_max = ptr[x];
            }
        }

        // 2. Apply min-max normalization formula
        // I_new = (I_old - I_min) / (I_max - I_min) * 255
        cv::Mat result(gray.size(), gray.type());
        
        if (I_max == I_min) {
            // Edge case: Image is completely flat (solid color)
            result = gray.clone();
        } else {
            float scale = 255.0f / (I_max - I_min);
            for (int y = 0; y < gray.rows; ++y) {
                const uchar* src_ptr = gray.ptr<uchar>(y);
                uchar* dst_ptr = result.ptr<uchar>(y);
                for (int x = 0; x < gray.cols; ++x) {
                    dst_ptr[x] = cv::saturate_cast<uchar>(std::round((src_ptr[x] - I_min) * scale));
                }
            }
        }

        // Convert back to BGR for consistent frontend display
        cv::Mat result_bgr;
        cv::cvtColor(result, result_bgr, cv::COLOR_GRAY2BGR);
        return result_bgr;
    }
};

// Pybind11 Wrappers
#include "binding_utils.h"

py::array_t<unsigned char> equalize_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = ImageEnhancer::equalizeHistogram(mat);
    return mat_to_numpy(res);
}

py::array_t<unsigned char> normalize_wrapper(py::array_t<unsigned char> img) {
    auto mat = numpy_to_mat(img);
    auto res = ImageEnhancer::normalizeImage(mat);
    return mat_to_numpy(res);
}

PYBIND11_MODULE(enhance_image, m) {
    m.doc() = "Image enhancement C++ backend";
    m.def("equalize", &equalize_wrapper, "Apply Histogram Equalization");
    m.def("normalize", &normalize_wrapper, "Apply Image Normalization");
}
