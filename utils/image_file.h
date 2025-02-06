#pragma once
#include "onnxruntime_c_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
int read_image_file(_In_z_ const ORTCHAR_T* input_file, _Out_ size_t* height, _Out_ size_t* width, _Outptr_ float** out,
                  _Out_ size_t* output_count);


int write_image_file(_In_ uint8_t* model_output_bytes, unsigned int height,
                     unsigned int width, _In_z_ const ORTCHAR_T* output_file);
#ifdef __cplusplus
}
#endif