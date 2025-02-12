#include "face_detector.h"

int detect_with_yoloface(const OrtApi *g_ort, OrtSession *session, const ORTCHAR_T *input_file, float **output_bounding_boxes, int *output_bounding_boxes_row, int *output_bounding_boxes_col, float **output_face_scores, int *output_face_scores_length, float **output_face_landmarks_5, int *output_face_landmarks_5_row, int *output_face_landmarks5_col)
{
  size_t input_height;
  size_t input_width;
  float *model_input;
  size_t model_input_ele_count;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0)
  {
    return -1;
  }

  for (int i = 0; i < model_input_ele_count; i++)
  {
    model_input[i] = (model_input[i] - 127.5) / 128.0;
  }

  OrtMemoryInfo *memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_shape[] = {1, 3, (int64_t)input_width, (int64_t)input_height};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  OrtValue *input_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                           input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &input_tensor));
  assert(input_tensor != NULL);
  int is_tensor;
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  const char *input_names[] = {"input"};
  const char *output_names[] = {"output"};
  OrtValue *output_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1, output_names, 1,
                                &output_tensor));
  assert(output_tensor != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  assert(is_tensor);
  int ret = 0;
  float *output_tensor_data = NULL;
  ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void **)&output_tensor_data));
  float *output_tensor_data_transpose = NULL;
  transpose(output_tensor_data, 20, 8400, (float **)&output_tensor_data_transpose);
  float *score_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &score_raw, 4, 1);
  int count = 0;
  int index[8400] = {0};
  for (int i = 0; i < 8400; i++)
  {
    double f = score_raw[i];
    if (f > 0.5)
    {
      index[count] = i;
      count++;
    }
  }
  *output_bounding_boxes_row = count;
  *output_face_scores_length = count;
  *output_face_landmarks_5_row = count;
  float *face_scores = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; i++)
  {
    face_scores[i] = (float)score_raw[index[i]];
  }
  /*
  for (int i = 0; i < count; i++) {
    qDebug() << face_scores[i];
  }
  */

  int bounding_box_length = 4;
  *output_bounding_boxes_col = bounding_box_length;
  float *bounding_box_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &bounding_box_raw, 0, bounding_box_length);
  float *bounding_box = (float *)malloc(count * bounding_box_length * sizeof(float));
  for (int j = 0; j < count; j++)
  {
    for (int i = 0; i < bounding_box_length; i++)
    {
      bounding_box[j * bounding_box_length + i] = (float)bounding_box_raw[index[j] * bounding_box_length + i];
    }
  }
  for (int j = 0; j < count; j++)
  {
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
  *output_face_landmarks5_col = face_landmark_5_length;
  float *face_landmark_5_raw = NULL;
  copy_partial_matrix(output_tensor_data_transpose, 8400, 20, &face_landmark_5_raw, 5, face_landmark_5_length);
  float *face_landmark_5 = (float *)malloc(count * face_landmark_5_length * sizeof(float));
  for (int j = 0; j < count; j++)
  {
    for (int i = 0; i < face_landmark_5_length; i++)
    {
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
  *output_bounding_boxes = bounding_box;
  *output_face_scores = face_scores;
  *output_face_landmarks_5 = face_landmark_5;
  free(model_input);
  free(output_tensor_data_transpose);
  free(score_raw);
  free(bounding_box_raw);
  free(face_landmark_5_raw);
  return ret;
}

void verify_input_output_count(OrtSession *session)
{
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

void detect_faces(const OrtApi *g_ort, OrtEnv *env, OrtSessionOptions *session_options, ORTCHAR_T *input_file, float **output_bounding_boxes, int *output_bounding_boxes_row, int *output_bounding_boxes_col, float **output_face_scores, int *output_face_scores_length, float **output_face_landmarks_5, int *output_face_landmarks_5_row, int *output_face_landmarks5_col)
{
  OrtSession *session;
  int ret = 0;
  ORTCHAR_T *model_path = "./models/yoloface_8n.onnx";

  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
  verify_input_output_count(session);
  ret = detect_with_yoloface(g_ort, session, input_file, output_bounding_boxes, output_bounding_boxes_row, output_bounding_boxes_col, output_face_scores, output_face_scores_length, output_face_landmarks_5, output_face_landmarks_5_row, output_face_landmarks5_col);
  g_ort->ReleaseSession(session);
  if (ret != 0)
  {
    fprintf(stderr, "fail\n");
  }
}