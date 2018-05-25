MAPLAB_COMMON="$MAPLAB/common/maplab-common/proto"
PROTO="$MAPLAB/algorithms/loopclosure/loop-closure-handler/proto/loop-closure-handler"

protoc -I=$MAPLAB_COMMON:$PROTO --python_out=./ $PROTO/debug_fusion.proto
