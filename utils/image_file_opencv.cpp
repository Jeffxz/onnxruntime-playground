#include "image_file.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

int read_image_file(const ORTCHAR_T* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  cv::Mat img = cv::imread(input_file);
  if (img.empty()) {
      return -1;
  }
  hwc_to_chw(img.data, img.rows, img.cols, out, output_count);
  *width = img.cols;
  *height = img.rows;
  return 0;
}

int write_image_file(uint8_t* model_output_bytes, unsigned int height,
                     unsigned int width, const ORTCHAR_T* output_file){
  return 0;
}
