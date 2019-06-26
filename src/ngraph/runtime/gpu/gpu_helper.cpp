#include "node.hpp"
#include "op/convolution.hpp"
#include "runtime/gpu/cudnn_descriptors.hpp"
#include "runtime/gpu/cudnn_emitter.hpp"
#include "runtime/gpu/gpu_backend.hpp"
#include "runtime/gpu/gpu_op_annotations.hpp"
#include "runtime/gpu/gpu_runtime_context.hpp"

#include "runtime/gpu/gpu_helper.hpp"

#include <cudnn.h>

using namespace std;
using namespace ngraph;


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
        const std::shared_ptr<Node> node
        //const std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx
        )
{
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        return get_algo_options(op_convolution);//, ctx);
    }

    return {};
}

// This follows the style of `build_convolution` in `cudnn_emitter.cpp`.
std::vector<std::tuple<size_t, float, size_t>> runtime::gpu::get_algo_options(
        const std::shared_ptr<op::Convolution> node
        //const std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx
        )
{
    // Get the backend context from the op annotations
    auto annotation = dynamic_pointer_cast<runtime::gpu::ConvFwdAnnotations>(node->get_op_annotations()); 
    NGRAPH_ASSERT(annotation);
    std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx = annotation->get_context();

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
