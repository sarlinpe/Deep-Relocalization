#ifndef DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
#define DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_

#include <memory>
#include <vector>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <vi-map/vi-map.h>
#include <vi-map/unique-id.h>

#include "deep-relocalization/tensorflow-net.h"
#include "deep-relocalization/kd-tree-index.h"
#include "deep-relocalization/pca-reduction.h"
#include "deep-relocalization/descriptor_index.pb.h"

class PlaceRetrieval {
  public:
    PlaceRetrieval(const std::string model_path);

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
    std::mutex network_mutex_;
    std::unique_ptr<KDTreeIndex> index_;
    std::unique_ptr<PcaReduction> pca_reduction_;
    vi_map::VisualFrameIdentifierList indexed_frame_identifiers_;
};

#endif  // DEEP_RELOCALIZATION_PLACE_RETRIEVAL_H_
