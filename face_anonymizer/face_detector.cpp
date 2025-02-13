#include "face_detector.h"

int detect_with_yoloface(const OrtApi *g_ort, OrtSession *session,
                         float *model_input, int input_width, int input_height,
                         int model_input_ele_count,
                         float **out_bounding_boxes,
                         float **out_face_scores,
                         float **out_face_landmarks_5, int *out_row) {
    for (int i = 0; i < model_input_ele_count; i++) {
        model_input[i] = (model_input[i] - 127.5) / 128.0;
    }

    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    const int64_t input_shape[] = {1, 3, (int64_t)input_width,
                                   (int64_t)input_height};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    const size_t model_input_len = model_input_ele_count * sizeof(float);

    OrtValue *input_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, model_input, model_input_len, input_shape, input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    assert(input_tensor != NULL);
    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    const char *input_names[] = {"input"};
    const char *out_names[] = {"output"};
    OrtValue *out_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names,
                                  (const OrtValue *const *)&input_tensor, 1,
                                  out_names, 1, &out_tensor));
    assert(out_tensor != NULL);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(out_tensor, &is_tensor));
    assert(is_tensor);
    int ret = 0;
    float *out_tensor_data = NULL;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(
        out_tensor, (void **)&out_tensor_data));
    float *out_tensor_data_transpose = NULL;
    transpose(out_tensor_data, 20, 8400,
              (float **)&out_tensor_data_transpose);
    float *score_raw = NULL;
    copy_partial_matrix(out_tensor_data_transpose, 8400, 20, &score_raw, 4,
                        1);
    int count = 0;
    int index[8400] = {0};
    for (int i = 0; i < 8400; i++) {
        double f = score_raw[i];
        if (f > 0.5) {
            index[count] = i;
            count++;
        }
    }
    *out_row = count;
    float *face_scores = (float *)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        face_scores[i] = (float)score_raw[index[i]];
    }
    /*
    for (int i = 0; i < count; i++) {
      qDebug() << face_scores[i];
    }
    */

    int bounding_box_length = 4;
    float *bounding_box_raw = NULL;
    copy_partial_matrix(out_tensor_data_transpose, 8400, 20,
                        &bounding_box_raw, 0, bounding_box_length);
    float *bounding_box =
        (float *)malloc(count * bounding_box_length * sizeof(float));
    for (int j = 0; j < count; j++) {
        for (int i = 0; i < bounding_box_length; i++) {
            bounding_box[j * bounding_box_length + i] =
                (float)bounding_box_raw[index[j] * bounding_box_length + i];
        }
    }
    for (int j = 0; j < count; j++) {
        float x1 = bounding_box[j * bounding_box_length + 0] -
                   bounding_box[j * bounding_box_length + 2] / 2;
        float y1 = bounding_box[j * bounding_box_length + 1] -
                   bounding_box[j * bounding_box_length + 3] / 2;
        float x2 = bounding_box[j * bounding_box_length + 0] +
                   bounding_box[j * bounding_box_length + 2] / 2;
        float y2 = bounding_box[j * bounding_box_length + 1] +
                   bounding_box[j * bounding_box_length + 3] / 2;
        bounding_box[j * bounding_box_length + 0] = fmin(x1, x2);
        bounding_box[j * bounding_box_length + 1] = fmin(y1, y2);
        bounding_box[j * bounding_box_length + 2] = fmax(x1, x2);
        bounding_box[j * bounding_box_length + 3] = fmax(y1, y2);
    }
    /*
    for (int i = 0; i < bounding_box_length; i++) {
      qDebug() << bounding_box[9 * 4 + i];
    }
    */
    int face_landmark_5_length = 15;
    float *face_landmark_5_raw = NULL;
    copy_partial_matrix(out_tensor_data_transpose, 8400, 20,
                        &face_landmark_5_raw, 5, face_landmark_5_length);
    float *face_landmark_5 =
        (float *)malloc(count * face_landmark_5_length * sizeof(float));
    for (int j = 0; j < count; j++) {
        for (int i = 0; i < face_landmark_5_length; i++) {
            face_landmark_5[j * face_landmark_5_length + i] =
                face_landmark_5_raw[index[j] * face_landmark_5_length + i];
        }
    }
    /*
    for (int i = 0; i < face_landmark_5_length; i++) {
      qDebug() << face_landmark_5[i];
    }
    */
    g_ort->ReleaseValue(out_tensor);
    g_ort->ReleaseValue(input_tensor);
    *out_bounding_boxes = bounding_box;
    *out_face_scores = face_scores;
    *out_face_landmarks_5 = face_landmark_5;
    free(model_input);
    free(out_tensor_data_transpose);
    free(score_raw);
    free(bounding_box_raw);
    free(face_landmark_5_raw);
    return ret;
}

void verify_input_out_count(OrtSession *session) {
    size_t count;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
    assert(count == 1);
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
    assert(count == 1);
}

void detect_faces(const OrtApi *g_ort, OrtEnv *env,
                  OrtSessionOptions *session_options, float *image_data,
                  int image_width, int image_height, int image_data_ele_count,
                  float **out_bounding_boxes, float **out_face_scores,
                  float **out_face_landmarks_5, int *out_row) {
    OrtSession *session;
    int ret = 0;
    ORTCHAR_T *model_path = "./models/yoloface_8n.onnx";

    ORT_ABORT_ON_ERROR(
        g_ort->CreateSession(env, model_path, session_options, &session));
    verify_input_out_count(session);
    ret = detect_with_yoloface(g_ort, session, image_data, image_width,
                               image_height, image_data_ele_count,
                               out_bounding_boxes, out_face_scores,
                               out_face_landmarks_5, out_row);
    g_ort->ReleaseSession(session);
    if (ret != 0) {
        fprintf(stderr, "fail\n");
    }
}