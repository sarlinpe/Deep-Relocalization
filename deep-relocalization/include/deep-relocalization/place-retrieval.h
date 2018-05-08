#ifndef DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
#define DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <vi-map/vi-map.h>
#include <vi-map/unique-id.h>

#include "deep-relocalization/tensorflow-net.h"
#include "deep-relocalization/kd-tree-index.h"
#include "deep-relocalization/descriptor_index.pb.h"

class PlaceRetrieval {
  public:
    PlaceRetrieval(const std::string model_path):
            network_(model_path, "image", "descriptor") {
        index_.reset(new KDTreeIndex(network_.descriptor_size()));
    }

    void BuildIndexFromMap(
            const vi_map::VIMap& map,
            deep_relocalization::proto::DescriptorIndex* proto_index);

    void LoadIndex(const deep_relocalization::proto::DescriptorIndex& proto_index);

    void RetrieveNearestNeighbors(
            const cv::Mat& input_image, const unsigned num_neighbors,
            const float max_distance,
            vi_map::VisualFrameIdentifierList* retrieved_frame_identifiers);

  private:
    TensorflowNet network_;
    std::unique_ptr<KDTreeIndex> index_;
    vi_map::VisualFrameIdentifierList indexed_frame_identifiers_;
};

#endif  // DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
