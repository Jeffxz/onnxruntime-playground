#include "face_helper.h"
#include <opencv2/opencv.hpp>
#include <QDebug>

using namespace cv;
using namespace dnn;

void apply_nms(float *bounding_boxes, float *face_scores,
               float *face_landmarks_5, int rows, float score_threshold,
               float nms_threshold, int **out_indices, int *out_indices_size) {
    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    for (int i = 0; i < rows; i++) {
        float x1 = bounding_boxes[i * 4];
        float y1 = bounding_boxes[i * 4 + 1];
        float x2 = bounding_boxes[i * 4 + 2];
        float y2 = bounding_boxes[i * 4 + 3];
        boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(face_scores[i]);
    }

    NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
    int size = indices.size();
    *out_indices = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
      *out_indices[i] = indices[i];
    }
}
