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

bool runtime::gpu::can_select_algo(const std::shared_ptr<Node> node)
{
    // Check if it's a convolution
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        return true;
    }
    return false;
}

/////
///// set_algo
/////

// Top level function
void runtime::gpu::set_algo(const std::shared_ptr<Node> node, size_t algo_enum, size_t workspace_size)
{
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        set_algo(op_convolution, algo_enum, workspace_size);
    }
}

void runtime::gpu::set_algo(const std::shared_ptr<op::Convolution> node, size_t algo_enum, size_t workspace_size)
{
    // Create the GPU op annotations
    cudnnConvolutionFwdAlgo_t algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo_enum);
    // Tensor -> Op_Annotation -> attach to node
    std::string tensor_name = node->get_name() + "_workspace";
    auto workspace_tensor = std::make_shared<descriptor::Tensor>(
            element::u8,
            ngraph::Shape({workspace_size}),
            tensor_name
            );

    auto annotation = std::make_shared<runtime::gpu::ConvFwdAnnotations>();
    annotation->set_algo(algo);
    annotation->set_workspace_tensor(workspace_tensor);
    node->set_op_annotations(annotation);
}

/////
///// get_algo_options
/////

// Need to pass te backend context for getting the cudnn handle
std::vector<std::tuple<size_t, float, size_t>> runtime::gpu::get_algo_options(
        const std::shared_ptr<Node> node,
        const std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx
        )
{
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        return get_algo_options(op_convolution, ctx);
    }

    return {};
}

// This follows the style of `build_convolution` in `cudnn_emitter.cpp`.
std::vector<std::tuple<size_t, float, size_t>> runtime::gpu::get_algo_options(
        const std::shared_ptr<op::Convolution> node,
        const std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx
        )
{
    // Unpack stuff
    cudnnHandle_t cudnn_handle = *ctx->m_runtime_context->cudnn_handle;
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    auto input_shape = args[0].get_shape();
    auto filter_shape = args[1].get_shape();
    auto output_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides();
    Strides window_movement_strides = node->get_window_movement_strides();
    Strides data_dilation_strides = node->get_data_dilation_strides();
    CoordinateDiff padding_below_diff = node->get_padding_below();
    CoordinateDiff padding_above_diff = node->get_padding_above();
    auto dtype = out[0].get_element_type().c_type_string();

    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }

    // Setup come cudnn stuff
    cudnnDataType_t data_type = runtime::gpu::get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto& tensor_desc_0 =
        runtime::gpu::tensor_descriptor_from_shape(input_shape, data_type, tensor_format);
    auto& tensor_desc_1 =
        tensor_descriptor_from_shape(output_shape, data_type, tensor_format);
    auto& filter_desc = runtime::gpu::get_cudnn_filter_descriptor(filter_shape, data_type, tensor_format);
    auto& conv_desc = runtime::gpu::get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type);
    cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    int num_algos;
    int max_algos = 0;
    CUDNN_SAFE_CALL(
        // MARK:: TODO: how to get the handle ...
        cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> results(max_algos);

    // Get a vector of results
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                         tensor_desc_0,
                                         filter_desc,
                                         conv_desc,
                                         tensor_desc_1,
                                         static_cast<int>(results.size()),
                                         &num_algos,
                                         results.data()));
    results.resize(num_algos);

    // Construct the return type
    std::vector<std::tuple<size_t, float, size_t>> return_vec;
    for (auto res: results)
    {
        auto tup = std::tuple<size_t, float, size_t>(
                static_cast<size_t>(res.algo),
                res.time,
                res.memory
                );
        return_vec.push_back(tup);
    }
    return return_vec;
}

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
