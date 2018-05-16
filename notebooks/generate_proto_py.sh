MAPLAB="$MAPLAB/common/maplab-common/proto"
PROTO="../deep-relocalization/proto/deep-relocalization"

protoc -I=$MAPLAB:$PROTO --python_out=./ $PROTO/descriptor_index.proto $MAPLAB/maplab-common/id.proto $MAPLAB/maplab-common/eigen.proto
