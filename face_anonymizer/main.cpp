#include <assert.h>
#include <stdio.h>

#include <QCoreApplication>
#include <QDebug>

#include "face_detector.h"
#include "face_analyzer.h"
#include "image_file.h"
#include "inference_manager.h"
#include "utils.h"

#define tcscmp strcmp

static void usage() { printf("usage: <input_file> [cpu|cuda|dml] \n"); }

int enable_cuda(OrtSessionOptions *session_options) {
    // OrtCUDAProviderOptions is a C struct. C programming language doesn't have
    // constructors/destructors.
    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to
    // zero.
    memset(&o, 0, sizeof(o));
    // But is zero a valid value for every variable? Not quite. It is not
    // guaranteed. In the other words: does every enum type contain zero? The
    // following line can be omitted because EXHAUSTIVE is mapped to zero in
    // onnxruntime_c_api.h.
    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    OrtStatus *onnx_status =
        g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
    if (onnx_status != NULL) {
        const char *msg = g_ort->GetErrorMessage(onnx_status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(onnx_status);
        return -1;
    }
    return 0;
}

#ifdef USE_DML
void enable_dml(OrtSessionOptions *session_options) {
    ORT_ABORT_ON_ERROR(
        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
}
#endif

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    if (argc < 2) {
        usage();
        return -1;
    }

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
        return -1;
    }
    ORTCHAR_T *input_file = argv[1];
    // By default it will try CUDA first. If CUDA is not available, it will run
    // all the things on CPU. But you can also explicitly set it to
    // DML(directml) or CPU(which means cpu-only).
    ORTCHAR_T *execution_provider = (argc >= 5) ? argv[4] : NULL;
    OrtEnv *env;
    ORT_ABORT_ON_ERROR(
        g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    assert(env != NULL);
    int ret = 0;
    OrtSessionOptions *session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

    if (execution_provider) {
        if (tcscmp(execution_provider, ORT_TSTR("cpu")) == 0) {
            // Nothing; this is the default
        } else if (tcscmp(execution_provider, ORT_TSTR("dml")) == 0) {
#ifdef USE_DML
            enable_dml(session_options);
#else
            puts("DirectML is not enabled in this build.");
            return -1;
#endif
        } else if (tcscmp(execution_provider, ORT_TSTR("cuda")) == 0) {
            printf("Try to enable CUDA first\n");
            ret = enable_cuda(session_options);
            if (ret) {
                fprintf(stderr, "CUDA is not available\n");
                return -1;
            } else {
                printf("CUDA is enabled\n");
            }
        }
    }

    size_t image_width = 0;
    size_t image_height = 0;
    float *image_data = NULL;
    size_t image_data_ele_count = 0;
    if (read_image_file(input_file, &image_height, &image_width, &image_data,
                        &image_data_ele_count) != 0) {
        return -1;
    }

    float *bounding_boxes;
    float *face_scores;
    float *face_landmarks_5;
    int row = 0;
    detect_faces(g_ort, env, session_options, image_data, image_width,
                 image_height, image_data_ele_count, &bounding_boxes,
                 &face_scores, &face_landmarks_5, &row);

    /*
    for (int i = 0; i < row; i++) {
        qDebug() << bounding_boxes[i * 4 + 0]
                 << bounding_boxes[i * 4 + 1]
                 << bounding_boxes[i * 4 + 2]
                 << bounding_boxes[i * 4 + 3];
    }
    */
    create_faces(g_ort, env, session_options, image_data, image_width,
                 image_height, bounding_boxes, face_scores, face_landmarks_5,
                 row);

    free(bounding_boxes);
    free(face_scores);
    free(face_landmarks_5);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
}
