#include <QDebug>
#include <assert.h>
#include <stdio.h>
#include "inference_manager.h"
#include "image_file.h"
#include "utils.h"

void detect_faces(const OrtApi *g_ort, OrtEnv *env, OrtSessionOptions *session_options, ORTCHAR_T *input_file, float **output_bounding_boxes, int *output_bounding_boxes_row, int *output_bounding_boxes_col, float **output_face_scores, int *output_face_scores_length, float **output_face_landmarks_5, int *output_face_landmarks_5_row, int *output_face_landmarks5_col);