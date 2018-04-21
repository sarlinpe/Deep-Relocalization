cmake_minimum_required(VERSION 2.8)
project(deep_relocalization)

find_package(catkin_simple REQUIRED)
catkin_simple()

# Protobufs
set(PROTO_DEFNS proto/deep-relocalization/descriptor_index.proto)
PROTOBUF_CATKIN_GENERATE_CPP2("proto" PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
cs_add_library(${PROJECT_NAME}_lib ${PROTO_SRCS} ${PROTO_HDRS})

# Preprocessor macros
add_definitions(-DDATA_ROOT_PATH="${CMAKE_SOURCE_DIR}/data")
add_definitions(-DMODEL_ROOT_PATH="${CMAKE_SOURCE_DIR}/model")

# Test files
cs_add_executable(test_opencv test/test_opencv.cc)
cs_add_executable(test_tensorflow test/test_tensorflow.cc)
cs_add_executable(test_protobuf test/test_protobuf.cc)
target_link_libraries(test_protobuf ${PROJECT_NAME}_lib)
cs_add_executable(test_inference test/test_inference.cc)


cs_install()
cs_export()

