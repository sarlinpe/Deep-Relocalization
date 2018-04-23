#include <string>
#include <iostream>

#include <glog/logging.h>

#include <vi-map/vi-map.h>
#include <posegraph/pose-graph.h>
#include <aslam/frames/visual-frame.h>
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
