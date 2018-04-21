#ifndef DEEP_RELOCALIZATION_TENSORFLOW_NET_H_
#define DEEP_RELOCALIZATION_TENSORFLOW_NET_H_

#include <vector>

#include <glog/logging.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

using tensorflow::Status;
using tensorflow::Tensor;

class TensorflowNet {
  public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> DescriptorType;

    TensorflowNet(const std::string model_path ,
                  const std::string input_tensor_name,
                  const std::string output_tensor_name):
                input_name(input_tensor_name), output_name(output_tensor_name) {
        CHECK(tensorflow::MaybeSavedModelDirectory(model_path));

        // Load model
        Status status = tensorflow::LoadSavedModel(
                tensorflow::SessionOptions(), tensorflow::RunOptions(),
                model_path, {tensorflow::kSavedModelTagServe}, &bundle);
        if (!status.ok())
            LOG(FATAL) << status.ToString();

        // Check input and output shapes
        tensorflow::GraphDef graph_def = bundle.meta_graph_def.graph_def();
        for(auto &node: graph_def.node()) {
            if(node.name() == input_tensor_name) {
                input_channels = node.attr().at("shape").shape().dim(3).size();
            }
            if(node.name() == output_tensor_name) {
                // Hack as the identity node does not have the shape attribute
                descriptor_size = node.attr().at(
                        "_output_shapes").list().shape(0).dim(1).size();
            }
        }
    }

    void perform_inference(cv::Mat& image, DescriptorType& descriptor) {
        CHECK(image.data);
        CHECK(image.isContinuous());
        CHECK(image.channels() == input_channels);

        int height = image.size().height, width = image.size().width;
        if(image.type() != CV_32F)
            image.convertTo(image, CV_32F);

        // Prepare input tensor
        Tensor input_tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({1, height, width, input_channels}));
        // TODO: avoid copy if possible
        tensorflow::StringPiece tmp_data = input_tensor.tensor_data();
        std::memcpy(const_cast<char*>(tmp_data.data()), image.data,
                    height * width * input_channels * sizeof(float));

        // Run inference
        std::vector<Tensor> outputs;
        Status status = bundle.session->Run({{input_name+":0", input_tensor}},
                                            {output_name+":0"}, {}, &outputs);
        if (!status.ok())
            LOG(FATAL) << status.ToString();

        // Copy result
        float *descriptor_ptr = outputs[0].flat<float>().data();
        Eigen::Map<DescriptorType> descriptor_map(descriptor_ptr, descriptor_size);
        descriptor = descriptor_map;  // Copy
    }

    int get_descriptor_size() {
        return descriptor_size;
    }

  private:
    tensorflow::SavedModelBundle bundle;
    std::string input_name;
    std::string output_name;
    int descriptor_size;
    int input_channels;
};

#endif  // DEEP_RELOCALIZATION_TENSORFLOW_NET_H_
