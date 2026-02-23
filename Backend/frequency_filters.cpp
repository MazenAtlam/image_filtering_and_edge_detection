private:
    // Helper function to rearrange the quadrants of Fourier image
    static void fftShift(cv::Mat& magI) {
        magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;

        cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
        cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

public:
    // Apply the Frequency domain Filters
    static cv::Mat applyFFTFilter(const cv::Mat& image, const std::string& filter_type, int radius) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Expand input image to optimal size for fast computation
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(gray.rows);
        int n = cv::getOptimalDFTSize(gray.cols);
        cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        // Make place for both the complex and the real values
        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);

        // Make the Discrete Fourier Transform
        cv::dft(complexI, complexI);

        // Shift quadrants to the center
        fftShift(complexI);

        // Create Mask (Low Pass or High Pass)
        cv::Mat mask = cv::Mat(complexI.size(), CV_32F, cv::Scalar(filter_type == "low_pass" ? 0 : 1));
        cv::Point center(complexI.cols / 2, complexI.rows / 2);
        cv::circle(mask, center, radius, cv::Scalar(filter_type == "low_pass" ? 1 : 0), -1);

        // Merge mask into two channels to match complexI
        cv::Mat maskPlanes[] = {mask, mask};
        cv::Mat complexMask;
        cv::merge(maskPlanes, 2, complexMask);

        // Apply mask (element-wise multiplication)
        cv::mulSpectrums(complexI, complexMask, complexI, 0);

        // Shift back
        fftShift(complexI);

        // Inverse Transform (IDFT)
        cv::idft(complexI, complexI, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        cv::split(complexI, planes);
        cv::Mat img_back = planes[0];

        // Crop back to original size and normalize
        img_back = img_back(cv::Rect(0, 0, gray.cols, gray.rows));
        cv::normalize(img_back, img_back, 0, 255, cv::NORM_MINMAX);
        
        cv::Mat result;
        img_back.convertTo(result, CV_8U);
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
        
        return result;
    }
