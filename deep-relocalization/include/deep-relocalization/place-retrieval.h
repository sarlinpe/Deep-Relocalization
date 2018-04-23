#ifndef DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
#define DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deep-relocalization/tensorflow-net.h"
#include "deep-relocalization/descriptor_index.pb.h"

class PlaceRetrieval {
  public:
    PlaceRetrieval(const std::string model_path):
        network(model_path, "image", "descriptor") {};


    void loadIndex(const std::string proto_path);  // TODO

    void retrieveNearest(const cv::Mat& input_image);  // TODO (with parameters)

    void BuildIndexFromMap(
            const vi_map::VIMap& map,
            deep_relocalization::proto::DescriptorIndex* proto_index);

  private:
    TensorflowNet network_;
};

#endif  // DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
