#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <assert.h>
#include <stdio.h>

#include <QDebug>

#include "image_file.h"
#include "inference_manager.h"
#include "utils.h"

void detect_faces(const OrtApi *g_ort, OrtEnv *env,
                  OrtSessionOptions *session_options, float *image_data,
                  int image_width, int image_height, int image_data_ele_count,
                  float **output_bounding_boxes, int *output_bounding_boxes_row,
                  int *output_bounding_boxes_col, float **output_face_scores,
                  int *output_face_scores_length,
                  float **output_face_landmarks_5,
                  int *output_face_landmarks_5_row,
                  int *output_face_landmarks5_col);

#endif