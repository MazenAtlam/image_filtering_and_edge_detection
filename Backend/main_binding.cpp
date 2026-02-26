#include <pybind11/pybind11.h>

#define FREQ_DONT_BIND
#include "get_intensity_data.cpp"
#include "adding_noise.cpp"
#include "filter_noise.cpp"
#include "edge_detection.cpp"
#include "enhance_image.cpp"
#include "frequency_filters.cpp"
#include "generate_hybrid.cpp"

namespace py = pybind11;

PYBIND11_MODULE(backend, m) {
    m.doc() = "Computer Vision Assignment 1 C++ Backend Module";

    // 1. Image I/O & Core Handling
    m.def("to_grayscale", &to_grayscale_wrapper, "Convert image to grayscale");
    m.def("calculate_histogram", &histogram_wrapper, "Calculate 256-bin histogram for each channel");
    m.def("calculate_cdf", &cdf_wrapper, "Calculate Cumulative Distribution Function for each channel");

    // 2. Additive Noise
    m.def("add_noise", &add_noise_wrapper, "Add noise to an image dynamically based on type and intensity",
          py::arg("image"), py::arg("noise_type"), py::arg("intensity"));

    // 3. Spatial Domain Filtering
    m.def("apply_filter", &apply_filter_wrapper, "Apply spatial filters based on type and kernel size",
          py::arg("image"), py::arg("filter_type"), py::arg("kernel_size"));

    // 4. Edge Detection
    m.def("canny", &canny_wrapper, "Apply Canny edge detection");
    m.def("sobel", &sobel_wrapper, "Apply Sobel edge detection");
    m.def("prewitt", &prewitt_wrapper, "Apply Prewitt edge detection");
    m.def("roberts", &roberts_wrapper, "Apply Roberts edge detection");

    // 5. Contrast Enhancement & Histograms
    m.def("equalize", &equalize_wrapper, "Apply Histogram Equalization");
    m.def("normalize", &normalize_wrapper, "Apply Image Normalization");

    // 6. Frequency Domain Filtering & Hybrid Images
    m.def("apply_fft", &apply_fft_wrapper, "Apply Low-pass or High-pass FFT filter",
          py::arg("image"), py::arg("filter_type"), py::arg("radius"));
    m.def("create_hybrid", &create_hybrid_wrapper, "Create hybrid image from two inputs",
          py::arg("img_a"), py::arg("img_b"), py::arg("radius_a"), py::arg("radius_b"));
}
