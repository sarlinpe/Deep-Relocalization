#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

int main () {
    string image_path = string(DATA_PATH) + "/nclt_sample.png";
    cout << image_path << endl;
    cv::Mat image;
    image = cv::imread(image_path, cv::IMREAD_ANYDEPTH);
    if(!image.data) {
        cout <<  " No image data." << endl;
        return -1;
    }
    cout << "Image size: " << image.size() << endl;
}
