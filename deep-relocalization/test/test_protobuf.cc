#include <iostream>

#include "deep-relocalization/descriptor_index.pb.h"

using namespace std;
using namespace deep_relocalization::proto;

int main () {
    DescriptorIndex index;
    index.set_model_name("test");
    index.set_descriptor_shape(3);

    for(int i = 0; i < 3; i++) {
        DescriptorIndex::KeyFrame* key_frame = index.add_key_frames();
        key_frame->set_id(i);
        DescriptorIndex::KeyFrame::Descriptor descriptor = key_frame->frame_descriptor();
        for(int j = 0; j < index.descriptor_shape(); j++) {
            descriptor.add_value(i+j);
        }
    }

    cout << index.DebugString() << endl;
}
