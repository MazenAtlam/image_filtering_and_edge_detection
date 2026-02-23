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
