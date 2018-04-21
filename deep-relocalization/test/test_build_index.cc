#include <iostream>
#include <string>

#include "deep-relocalization/descriptor_index.pb.h"
#include "deep-relocalization/place-retrieval.h"

using namespace std;

int main () {
    string map_path = string(MAP_ROOT_PATH) + "euroc_ml1";
    string model_path = string(MODEL_ROOT_PATH)
                        + "resnet50_delf_vlad_triplets_margin-02_proj-40_sq/";

    deep_relocalization::proto::DescriptorIndex proto_index;

    PlaceRetrieval retrieval(model_path);
    retrieval.buildIndexFromMap(map_path, &proto_index);

    cout << "Processed " << proto_index.frames_size() << " frames." << endl;
}
