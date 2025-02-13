#ifndef FACE_HELPER_H
#define FACE_HELPER_H

void apply_nms(float *bounding_boxes, float *face_scores,
               float *face_landmarks_5, int rows, float score_threshold,
               float nms_threshold, int **out_indices);

#endif