#include <QCoreApplication>
#include <QDebug>
#include <assert.h>
#include <stdio.h>
#include "onnxruntime_c_api.h"
#include "image_file.h"
#include "utils.h"

#define tcscmp strcmp

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

static void usage() { printf("usage: <model_path> <input_file> <output_file> [cpu|cuda|dml] \n"); }

int run_inference(OrtSession* session, const ORTCHAR_T* input_file) {
  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }

  for (int i = 0; i < model_input_ele_count; i++) {
    model_input[i] = (model_input[i] - 127.5) / 128.0;
  }

  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, 3, (int64_t)input_width, (int64_t)input_height};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue* input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  const char* input_names[] = {"input"};
  const char* output_names[] = {"output"};
  OrtValue* output_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
                                &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  int ret = 0;
  float* output_tensor_data = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
  float* output_tensor_data_transpose = NULL;
  transpose(output_tensor_data, 20, 8400, (float**)&output_tensor_data_transpose);
  float* score_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &score_raw, 4, 1);
  int count = 0;
  int index[8400] = {0};
  for (int i = 0; i < 8400; i++) {
    double f = score_raw[i];
    if (f > 0.5) {
      index[count] = i;
      count++;
    }
  }
  int bounding_box_length = 4;
  float* bounding_box_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &bounding_box_raw, 0, bounding_box_length);
  float* bounding_box = (float*)malloc(count * bounding_box_length * sizeof(float));
  for (int j = 0; j < count; j++) {
    for (int i = 0; i < bounding_box_length; i++) {
      bounding_box[j * bounding_box_length + i] = (float)bounding_box_raw[index[j] * bounding_box_length + i];
    }
  }
  for (int j = 0; j < count; j++) {
    float x1 = bounding_box[j * bounding_box_length + 0] - bounding_box[j * bounding_box_length + 2] / 2;
    float y1 = bounding_box[j * bounding_box_length + 1] - bounding_box[j * bounding_box_length + 3] / 2;
    float x2 = bounding_box[j * bounding_box_length + 0] + bounding_box[j * bounding_box_length + 2] / 2;
    float y2 = bounding_box[j * bounding_box_length + 1] + bounding_box[j * bounding_box_length + 3] / 2;
    bounding_box[j * bounding_box_length + 0] = x1;
    bounding_box[j * bounding_box_length + 1] = y1;
    bounding_box[j * bounding_box_length + 2] = x2;
    bounding_box[j * bounding_box_length + 3] = y2;
  }
  /*
  for (int i = 0; i < bounding_box_length; i++) {
    qDebug() << bounding_box[9 * 4 + i];
  }
  */
  int face_landmark_5_length = 15;
  float* face_landmark_5_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &face_landmark_5_raw, 5, face_landmark_5_length);
  float* face_landmark_5 = (float*)malloc(count * face_landmark_5_length * sizeof(float));
  for (int j = 0; j < count; j++) {
    for (int i = 0; i < face_landmark_5_length; i++) {
      face_landmark_5[j * face_landmark_5_length + i] = face_landmark_5_raw[index[j] * face_landmark_5_length + i];
    }
  }
  /*
  for (int i = 0; i < face_landmark_5_length; i++) {
    qDebug() << face_landmark_5[i];
  }
  */
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  free(model_input);
  free(output_tensor_data_transpose);
  free(bounding_box_raw);
  free(bounding_box);
  free(face_landmark_5_raw);
  free(face_landmark_5);
  return ret;
}

void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

int enable_cuda(OrtSessionOptions* session_options) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
  // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char* msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}

#ifdef USE_DML
void enable_dml(OrtSessionOptions* session_options) {
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
}
#endif

int main(int argc, char* argv[]) {
  QCoreApplication app(argc, argv);
  if (argc < 3) {
    usage();
    return -1;
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
    return -1;
  }
  ORTCHAR_T* model_path = argv[1];
  ORTCHAR_T* input_file = argv[2];
  // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
  // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
  ORTCHAR_T* execution_provider = (argc >= 5) ? argv[4] : NULL;
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);
  int ret = 0;
  OrtSessionOptions* session_options;
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

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
  verify_input_output_count(session);
  ret = run_inference(session, input_file);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);
  if (ret != 0) {
    fprintf(stderr, "fail\n");
  }
  return ret;
}
