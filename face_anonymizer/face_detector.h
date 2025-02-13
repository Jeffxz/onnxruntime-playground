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
                  float **output_bounding_boxes, float **output_face_scores,
                  float **output_face_landmarks_5,
                  int *output_row);

#endif