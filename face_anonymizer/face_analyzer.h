#ifndef FACE_ANALYZER_H
#define FACE_ANALYZER_H

#include <assert.h>
#include <stdio.h>

#include <QDebug>

#include "image_file.h"
#include "inference_manager.h"
#include "utils.h"

void create_faces(const OrtApi *g_ort, OrtEnv *env,
                  OrtSessionOptions *session_options, float *image_data,
                  int image_width, int image_height, float *bounding_boxes,
                  int bounding_boxes_row, int bounding_boxes_col,
                  float *face_scores, int face_scores_length,
                  float *face_landmarks_5, int face_landmarks_5_row,
                  int face_landmarks5_col);

#endif
