#include <assert.h>
#include <stdio.h>

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count);

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
void chw_to_hwc(const float* input, size_t h, size_t w, uint8_t** output);

void transpose(const float* input, size_t h, size_t w, float** output);

void copy_partial_matrix(const float* input, size_t h, size_t w, float** output, int offset_start, int length);