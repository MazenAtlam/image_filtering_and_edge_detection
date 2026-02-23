// 2. Make the Hybrid Image
    static cv::Mat createHybridImage(const cv::Mat& img_a, const cv::Mat& img_b, int radius_a, int radius_b) {
        cv::Mat b_resized;
        
        // Ensure images are the same size
        if (img_a.size() != img_b.size()) {
            cv::resize(img_b, b_resized, img_a.size());
        } else {
            b_resized = img_b.clone();
        }

        // Apply Low-pass to Image A and High-pass to Image B
        cv::Mat low_pass_a = applyFFTFilter(img_a, "low_pass", radius_a);
        cv::Mat high_pass_b = applyFFTFilter(b_resized, "high_pass", radius_b);

        // Combine them
        cv::Mat hybrid;
        cv::add(low_pass_a, high_pass_b, hybrid);
        
        return hybrid;
    }
