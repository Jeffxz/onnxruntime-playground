#include "face_analyzer.h"

#include <QDebug>

#include "face_helper.h"

void create_faces(const OrtApi *g_ort, OrtEnv *env,
                  OrtSessionOptions *session_options, float *image_data,
                  int image_width, int image_height, float *bounding_boxes,
                  float *face_scores, float *face_landmarks_5, int row) {
    int *keep_indices;
    int indices_size;
    apply_nms(bounding_boxes, face_scores, face_landmarks_5, row, 0.5, 0.4,
              &keep_indices, &indices_size);
    /*
      qDebug() << "indices size" << indices_size;
      for (int i = 0; i < indices_size; i++) {
      qDebug() << "indices " << i << keep_indices[i];
      }
    */

    free(keep_indices);
}
