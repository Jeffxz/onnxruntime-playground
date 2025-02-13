#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include "onnxruntime_c_api.h"

extern const OrtApi* g_ort;

#define ORT_ABORT_ON_ERROR(expr)                                   \
    do {                                                           \
        OrtStatus* onnx_status = (expr);                           \
        if (onnx_status != NULL) {                                 \
            const char* msg = g_ort->GetErrorMessage(onnx_status); \
            fprintf(stderr, "%s\n", msg);                          \
            g_ort->ReleaseStatus(onnx_status);                     \
            abort();                                               \
        }                                                          \
    } while (0);

#endif