#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// --- 1. Your Core C++ Logic ---
// This could be heavy math, image processing, or data analysis.
std::string process_data(const std::string& input_text) {
    // We are just modifying the string to prove C++ touched it
    return "C++ Processed: [" + input_text + "] successfully!";
}

// --- 2. The pybind11 Wrapper ---
// This block creates the bridge. It tells Python what it is allowed to see.
PYBIND11_MODULE(my_backend, m) {
    m.doc() = "C++ backend module for PyQt"; // Optional module docstring
    
    // Expose the process_data function to Python
    m.def("process_data", &process_data, "A function that processes a string");
}