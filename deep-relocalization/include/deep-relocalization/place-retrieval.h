#ifndef DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
#define DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>

#include "deep-relocalization/tensorflow-net.h"
#include "deep-relocalization/descriptor_index.pb.h"

class PlaceRetrieval {
  public:
    PlaceRetrieval(const std::string model_path):
        network(model_path, "image", "descriptor") {};

    void buildIndexFromMap(const std::string map_path,
        deep_relocalization::proto::DescriptorIndex* proto_index);

    void loadIndex(const std::string proto_path);  // TODO

    void retrieveNearest(const cv::Mat& input_image);  // TODO (with parameters)

  private:
    TensorflowNet network;
};

#endif  // DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
