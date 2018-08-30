#include <iostream>
#include <string>

#include <vi-map/vi-map.h>

#include "deep-relocalization/descriptor_index.pb.h"
#include "deep-relocalization/place-retrieval.h"

using namespace std;

int main () {
    string map_path = string(MAP_ROOT_PATH)
        + "lindenhof_afternoon-wet_aligned_deep-reloc_ext-int-3/";
    string model_path = string(MODEL_ROOT_PATH) + "mobilenetvlad_depth-0.35";

    deep_relocalization::proto::DescriptorIndex proto_index;

    vi_map::VIMap map;
    CHECK(map.loadFromFolder(map_path)) << "Loading of the vi-map failed.";

    PlaceRetrieval retrieval(model_path);
    retrieval.BuildIndexFromMap(map, &proto_index);

    cout << "Processed " << proto_index.frames_size() << " frames." << endl;
}
