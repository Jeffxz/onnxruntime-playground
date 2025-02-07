#include "utils.h"

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
void chw_to_hwc(const float* input, size_t h, size_t w, uint8_t** output) {
  size_t stride = h * w;
  uint8_t* output_data = (uint8_t*)malloc(stride * 3);
  assert(output_data != NULL);
  for (size_t c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f) f = 0;
      output_data[i * 3 + c] = (uint8_t)f;
    }
  }
  *output = output_data;
}

void transpose(const float* input, size_t h, size_t w, float** output) {
  size_t size = h * w;
  float* output_data = (float*)malloc(size * sizeof(float));
  for (size_t j = 0; j < h; j++) {
    for (size_t i = 0; i < w; i++) {
      output_data[i * h + j] = input[j * w + i];
    }
  }
  *output = output_data;
}

void copy_partial_matrix(const float* input, size_t h, size_t w, float** output, int offset_start, int length) {
  size_t size = h * length;
  float* output_data = (float*)malloc(size * sizeof(float));
  size_t offset_end = offset_start + length;
  for (size_t j = 0; j < h; j++) {
    for (size_t i = offset_start; i < offset_end; i++) {
      output_data[j * length + i - offset_start] = input[j * w + i];
    }
  }
  *output = output_data;
}