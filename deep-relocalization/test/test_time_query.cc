#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>

#include <vi-map/unique-id.h>
#include <aslam/common/timer.h>
#include <maplab-common/progress-bar.h>
#include <map-resources/resource-common.h>

#include "deep-relocalization/descriptor_index.pb.h"
#include "deep-relocalization/place-retrieval.h"

using namespace std;

int main () {
    string model_name = "resnet50_delf_vlad_triplets_margin-03_sq-in";
    string index_name = "lindenhof_afternoon_aligned_model-nopca.pb";
    string query_map_name = "lindenhof_afternoon-wet_aligned";
    string query_mission_id = "f6837cac0168580aa8a66be7bbb20805";

    string model_path = string(MODEL_ROOT_PATH) + model_name;
    string index_path = string(DATA_ROOT_PATH) + index_name;
    string query_map_path = string(MAP_ROOT_PATH) + query_map_name;

    PlaceRetrieval retrieval(model_path);

    // Load index
    deep_relocalization::proto::DescriptorIndex proto_index;
    fstream input(index_path, ios::in | ios::binary);
    CHECK(proto_index.ParseFromIstream(&input));
    CHECK_EQ(model_name, proto_index.model_name());
    retrieval.LoadIndex(proto_index);

    // Load query map
    vi_map::VIMap map;
    CHECK(map.loadFromFolder(query_map_path)) << "Loading of the vi-map failed.";
    vi_map::MissionId query_mission;
    map.ensureMissionIdValid(query_mission_id, &query_mission);
    CHECK(query_mission.isValid());

    // Get query vertices
    pose_graph::VertexIdList vertex_ids;
    map.getAllVertexIdsInMissionAlongGraph(query_mission, &vertex_ids);
    CHECK(!vertex_ids.empty());

    unsigned num_queries = 0, num_retrieved = 0;

    common::ProgressBar progress_bar(vertex_ids.size());
    for (const pose_graph::VertexId& vertex_id: vertex_ids) {
        progress_bar.increment();

        const vi_map::Vertex& vertex = map.getVertex(vertex_id);
        if(!vertex.numFrames())
            continue;
        unsigned frame_index = 0;
        backend::ResourceIdSet resource_ids;
        vertex.getFrameResourceIdsOfType(
                frame_index, backend::ResourceType::kRawImage, &resource_ids);
        if(!resource_ids.size())
            continue;
        cv::Mat image;
        CHECK(map.getRawImage(vertex, frame_index, &image));

        vi_map::VisualFrameIdentifierList retrieved_frames;
        retrieval.RetrieveNearestNeighbors(image, 20, 1, &retrieved_frames);

        ++num_queries;
        num_retrieved += retrieved_frames.size();
    }

    LOG(INFO) << "Average num of retrieved: "
        << static_cast<float>(num_retrieved) / num_queries;
    timing::Timing::Print(cout);
}
