#include <string>

#include <glog/logging.h>

#include <vi-map/vi-map.h>
#include <vi-map/unique-id.h>
#include <posegraph/pose-graph.h>
#include <posegraph/unique-id.h>
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

            unsigned frame_index = 0;  // Add only the first frame
            deep_relocalization::proto::DescriptorIndex::Frame* proto_frame =
                    proto_index->add_frames();
            vertex_id.serialize(proto_frame->mutable_vertex_id());
            proto_frame->set_frame_index(frame_index);

            // Compute and export descriptor
            cv::Mat image;
            map.getRawImage(vertex, frame_index, &image);
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
        pose_graph::VertexId vertex_id;
        vertex_id.deserialize(proto_frame.vertex_id());
        size_t frame_index = proto_frame.frame_index(); // static_cast<size_t>()
        indexed_frame_identifiers_.emplace_back(vertex_id, frame_index);

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
        const float max_distance,
        vi_map::VisualFrameIdentifierList* retrieved_frame_identifiers) {
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

    for (unsigned nn_search_idx = 0; nn_search_idx < num_neighbors; ++nn_search_idx) {
        const int nn_database_idx = indices(nn_search_idx, 0);
        const float nn_distance = distances(nn_search_idx, 0);
        if (nn_database_idx == -1 ||
                nn_distance == std::numeric_limits<float>::infinity()) {
            break;  // No more results
        }
        retrieved_frame_identifiers->push_back(
                indexed_frame_identifiers_[nn_database_idx]);
    }
}
