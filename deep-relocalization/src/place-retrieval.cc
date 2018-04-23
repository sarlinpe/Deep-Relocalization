#include <string>

#include <glog/logging.h>

#include <vi-map/vi-map.h>
#include <posegraph/pose-graph.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/common/unique-id.h>
#include <aslam/common/timer.h>
#include <maplab-common/aslam-id-proto.h>
#include <maplab-common/eigen-proto.h>
#include <maplab-common/progress-bar.h>

#include <opencv2/opencv.hpp>

#include "deep-relocalization/place-retrieval.h"

void PlaceRetrieval::BuildIndexFromMap(
        const vi_map::VIMap& map,
        deep_relocalization::proto::DescriptorIndex* proto_index) {
    CHECK_NOTNULL(proto_index);
    proto_index->set_descriptor_size(network_.descriptor_size());

    vi_map::MissionIdList mission_ids;
    map.getAllMissionIdsSortedByTimestamp(&mission_ids);
    for (const vi_map::MissionId& mission_id : mission_ids) {
        pose_graph::VertexIdList vertex_ids;
        map.getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);

        LOG(INFO) << "Computing descriptors for mission #" << mission_id;
        common::ProgressBar progress_bar(vertex_ids.size());
        for (const pose_graph::VertexId& vertex_id : vertex_ids) {
            const vi_map::Vertex& vertex = map.getVertex(vertex_id);
            if(!vertex.numFrames())
                continue;

            // Add only the first frame
            unsigned frame_idx = 0;
            const aslam::VisualFrame& visual_frame = vertex.getVisualFrame(frame_idx);
            deep_relocalization::proto::DescriptorIndex::Frame* proto_frame =
                    proto_index->add_frames();
            common::aslam_id_proto::serialize(
                    visual_frame.getId(), proto_frame->mutable_id());

            // Compute and export descriptor
            cv::Mat image;
            map.getRawImage(vertex, frame_idx, &image);
            TensorflowNet::DescriptorType descriptor;
            descriptor.resize(network_.descriptor_size(), Eigen::NoChange);
            network_.PerformInference(image, &descriptor);
            common::eigen_proto::serialize(
                    Eigen::MatrixXf(descriptor),
                    proto_frame->mutable_global_descriptor());

            progress_bar.increment();
        }
    }
}

void PlaceRetrieval::LoadIndex(
        const deep_relocalization::proto::DescriptorIndex& proto_index) {
    CHECK_EQ(proto_index.descriptor_size(), network_.descriptor_size());

    LOG(INFO) << "Loading " << proto_index.frames_size()
              << " reference descriptors into index.";
    common::ProgressBar progress_bar(proto_index.frames_size());
    for (const deep_relocalization::proto::DescriptorIndex::Frame& proto_frame :
            proto_index.frames()) {
        indexed_frame_ids_.emplace_back();
        common::aslam_id_proto::deserialize(
                proto_frame.id(), &indexed_frame_ids_.back());

        KDTreeIndex::DescriptorMatrixType descriptor;
        common::eigen_proto::deserialize(proto_frame.global_descriptor(), &descriptor);
        CHECK_EQ(descriptor.rows(), network_.descriptor_size());
        index_->AddDescriptors(descriptor);

        progress_bar.increment();
    }
    index_->RefreshIndex();
}

void PlaceRetrieval::RetrieveNearestNeighbors(
        const cv::Mat& input_image, const unsigned num_neighbors,
        const float max_distance, FrameIdsType* retrieved_ids) {
    TensorflowNet::DescriptorType descriptor;
    descriptor.resize(network_.descriptor_size(), Eigen::NoChange);
    timing::Timer timer_inference("Deep Relocalization: Compute descriptor");
    network_.PerformInference(input_image, &descriptor);
    timer_inference.Stop();

    Eigen::MatrixXi indices;
    indices.resize(num_neighbors, 1);
    Eigen::MatrixXf distances;
    distances.resize(num_neighbors, 1);
    timing::Timer timer_get_nn("Deep Relocalization: Get neighbors");
    index_->GetNNearestNeighbors(
            descriptor, num_neighbors, &indices, &distances, max_distance);
    timer_get_nn.Stop();

    for (int nn_search_idx = 0; nn_search_idx < num_neighbors; ++nn_search_idx) {
        const int nn_database_idx = indices(nn_search_idx, 0);
        const float nn_distance = distances(nn_search_idx, 0);
        if (nn_database_idx == -1 ||
                nn_distance == std::numeric_limits<float>::infinity()) {
            break;  // No more results
        }
        retrieved_ids->push_back(indexed_frame_ids_[nn_database_idx]);
    }
}
