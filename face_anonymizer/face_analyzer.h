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
                  float *face_scores, float *face_landmarks_5, int row);

#endif
