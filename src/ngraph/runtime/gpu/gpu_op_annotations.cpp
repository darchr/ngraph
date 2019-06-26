#include "node.hpp"
#include "op/convolution.hpp"
#include "runtime/gpu/cudnn_descriptors.hpp"
#include "runtime/gpu/cudnn_emitter.hpp"
#include "runtime/gpu/gpu_backend.hpp"
#include "runtime/gpu/gpu_op_annotations.hpp"
#include "runtime/gpu/gpu_runtime_context.hpp"

#include <cudnn.h>

using namespace ngraph;
using namespace std;


/////
///// cudnn stuff
/////

cudnnTensorDescriptor_t& runtime::gpu::tensor_descriptor_from_shape(
    const Shape& shape, const cudnnDataType_t data_type, const cudnnTensorFormat_t tensor_format)
{
    auto descriptors = make_shared<runtime::gpu::CUDNNDescriptors>();
    cudnnTensorDescriptor_t& desc = descriptors->build<cudnnTensorDescriptor_t>();
    if (shape.size() < 4)
    {
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(shape[i]);
        }
        CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(desc,
                                                   tensor_format,
                                                   data_type,
                                                   dimensions[0],
                                                   dimensions[1],
                                                   dimensions[2],
                                                   dimensions[3]));
    }
    else if (shape.size() == 4)
    {
        CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(desc,
                                                   tensor_format,
                                                   data_type,
                                                   static_cast<int>(shape[0]),
                                                   static_cast<int>(shape[1]),
                                                   static_cast<int>(shape[2]),
                                                   static_cast<int>(shape[3])));
    }
    else
    {
        std::vector<int> dimensions(shape.size());
        for (auto i = 0u; i < shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(shape[i]);
        }
        CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            static_cast<int>(dimensions.size()),
            dimensions.data(),
            runtime::gpu::cudnn_util::compute_strides(dimensions).data()));
    }

    return desc;
}

cudnnDataType_t runtime::gpu::get_cudnn_datatype(std::string dtype)
{
    static const std::unordered_map<std::string, cudnnDataType_t> datatype_map{
        {"float", CUDNN_DATA_FLOAT},
        {"double", CUDNN_DATA_DOUBLE},
        {"int8_t", CUDNN_DATA_INT8},
        {"int32_t", CUDNN_DATA_INT32}};
    auto p = datatype_map.find(dtype);
    if (p == datatype_map.end())
    {
        std::string err = dtype + "is not supported by cuDNN";
        throw std::runtime_error(err);
    }
    return p->second;
}

cudnnDataType_t runtime::gpu::get_cudnn_datatype(const element::Type& dtype)
{
    return get_cudnn_datatype(dtype.c_type_string());
}

cudnnFilterDescriptor_t& runtime::gpu::get_cudnn_filter_descriptor(
    const Shape& shape, const cudnnDataType_t data_type, const cudnnTensorFormat_t tensor_format)
{
    auto descriptors = make_shared<runtime::gpu::CUDNNDescriptors>();
    std::vector<int> dimensions(fmax(4, shape.size()), 1);
    int idx = 0;
    for (size_t i = dimensions.size() - shape.size(); i < dimensions.size(); i++)
    {
        dimensions[i] = static_cast<int>(shape[idx++]);
    }

    auto& filter_descriptor = descriptors->build<cudnnFilterDescriptor_t>();

    if (dimensions.size() <= 4)
    {
        CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_descriptor,
                                                   /*dataType=*/data_type,
                                                   /*format=*/tensor_format,
                                                   /*dimension_size*/ dimensions[0],
                                                   /*dimension_size*/ dimensions[1],
                                                   /*dimension_size*/ dimensions[2],
                                                   /*dimension_size*/ dimensions[3]));
    }
    else
    {
        CUDNN_SAFE_CALL(
            cudnnSetFilterNdDescriptor(filter_descriptor,
                                       /*dataType=*/data_type,
                                       /*format=*/tensor_format,
                                       /*num_dimensions=*/static_cast<int>(dimensions.size()),
                                       /*dimensions*/ dimensions.data()));
    }
    return filter_descriptor;
}

cudnnConvolutionDescriptor_t& runtime::gpu::get_cudnn_convolution_descriptor(
    const Shape& padding,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t data_type)
{

    auto descriptors = make_shared<runtime::gpu::CUDNNDescriptors>();
    auto& conv_descriptor = descriptors->build<cudnnConvolutionDescriptor_t>();
    std::vector<int> window_movement_strides_int(window_movement_strides.size());
    std::vector<int> window_dilation_strides_int(window_dilation_strides.size());
    std::vector<int> padding_int(padding.size());
    for (int i = 0; i < padding.size(); i++)
    {
        window_movement_strides_int[i] = static_cast<int>(window_movement_strides[i]);
        window_dilation_strides_int[i] = static_cast<int>(window_dilation_strides[i]);
        padding_int[i] = static_cast<int>(padding[i]);
    }

    if (padding.size() == 2)
    {
        CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                        padding_int[0],
                                                        padding_int[1],
                                                        window_movement_strides_int[0],
                                                        window_movement_strides_int[1],
                                                        window_dilation_strides_int[0],
                                                        window_dilation_strides_int[1],
                                                        mode,
                                                        data_type));
    }
    else
    {
        CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(conv_descriptor,
                                                        static_cast<int>(padding_int.size()),
                                                        padding_int.data(),
                                                        window_movement_strides_int.data(),
                                                        window_dilation_strides_int.data(),
                                                        mode,
                                                        data_type));
    }
    return conv_descriptor;
}
