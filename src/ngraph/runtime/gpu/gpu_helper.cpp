#include "runtime/gpu/gpu_helper.hpp"
#include <cudnn.h>

#include "node.hpp"
#include "op/convolution.hpp"
#include "runtime/gpu/cudnn_descriptors.hpp"
#include "runtime/gpu/cudnn_emitter.hpp"
#include "runtime/gpu/gpu_backend.hpp"
#include "runtime/gpu/gpu_op_annotations.hpp"
#include "runtime/gpu/gpu_runtime_context.hpp"

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

    auto op_convolution_bwd_data = 
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(node);
    if (op_convolution_bwd_data)
    {
        return true;
    }

    auto op_convolution_bwd_filters = 
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropFilters>(node);
    if (op_convolution_bwd_filters)
    {
        return true;
    }
    return false;
}

size_t runtime::gpu::get_workspace_tensor_offset(const std::shared_ptr<Node> node)
{
    if (runtime::gpu::has_algo(node.get()))
    {
        std::shared_ptr<ngraph::descriptor::Tensor> tensor = 
            runtime::gpu::get_workspace_tensor(node.get());

        return tensor->get_pool_offset();
    } else {
        return 0;
    }
}

size_t runtime::gpu::get_workspace_tensor_size(const std::shared_ptr<Node> node)
{
    if (runtime::gpu::has_algo(node.get()))
    {
        std::shared_ptr<ngraph::descriptor::Tensor> tensor = 
            runtime::gpu::get_workspace_tensor(node.get());

        return tensor->size();
    } else {
        return 0;
    }
}

/////
///// set_algo
/////

// Top level function
void runtime::gpu::set_algo(const std::shared_ptr<Node> node, size_t algo_enum, size_t workspace_size)
{
    // Convolution Forward
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        set_algo(op_convolution, algo_enum, workspace_size);
    }

    // Convolution Backprop Data
    auto op_convolution_bwd_data = std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(node); 
    if (op_convolution_bwd_data)
    {
        set_algo(op_convolution_bwd_data, algo_enum, workspace_size);
    }

    // Convolution Backprop Filters
    auto op_convolution_bwd_filters = 
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropFilters>(node); 
    if (op_convolution_bwd_filters)
    {
        set_algo(op_convolution_bwd_filters, algo_enum, workspace_size);
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

    auto annotation = std::dynamic_pointer_cast<runtime::gpu::ConvFwdAnnotations>(node->get_op_annotations());
    annotation->set_algo(algo);
    annotation->set_workspace_tensor(workspace_tensor);
}

void runtime::gpu::set_algo(const std::shared_ptr<op::ConvolutionBackpropData> node, size_t algo_enum, size_t workspace_size)
{
    // Create the GPU op annotations
    cudnnConvolutionBwdDataAlgo_t algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(algo_enum);
    // Tensor -> Op_Annotation -> attach to node
    std::string tensor_name = node->get_name() + "_workspace";
    auto workspace_tensor = std::make_shared<descriptor::Tensor>(
            element::u8,
            ngraph::Shape({workspace_size}),
            tensor_name
            );

    auto annotation = std::dynamic_pointer_cast<runtime::gpu::ConvBwdDataAnnotations>(node->get_op_annotations());
    annotation->set_algo(algo);
    annotation->set_workspace_tensor(workspace_tensor);
}

void runtime::gpu::set_algo(
        const std::shared_ptr<op::ConvolutionBackpropFilters> node, 
        size_t algo_enum, 
        size_t workspace_size)
{
    // Create the GPU op annotations
    cudnnConvolutionBwdFilterAlgo_t algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo_enum);
    // Tensor -> Op_Annotation -> attach to node
    std::string tensor_name = node->get_name() + "_workspace";
    auto workspace_tensor = std::make_shared<descriptor::Tensor>(
            element::u8,
            ngraph::Shape({workspace_size}),
            tensor_name
            );

    auto annotation = std::dynamic_pointer_cast<runtime::gpu::ConvBwdFilterAnnotations>(node->get_op_annotations());
    annotation->set_algo(algo);
    annotation->set_workspace_tensor(workspace_tensor);
}

/////
///// get_algo_options
/////

// Need to pass te backend context for getting the cudnn handle
std::vector<std::tuple<uint32_t, float, size_t, bool>> runtime::gpu::get_algo_options(
        const std::shared_ptr<Node> node)
{
    auto op_convolution = std::dynamic_pointer_cast<ngraph::op::Convolution>(node);
    if (op_convolution)
    {
        return get_algo_options(op_convolution);
    }
    auto op_convolution_bwd_data = std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropData>(node);
    if (op_convolution_bwd_data)
    {
        return get_algo_options(op_convolution_bwd_data);
    }

    auto op_convolution_bwd_filters = 
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBackpropFilters>(node);
    if (op_convolution_bwd_filters)
    {
        return get_algo_options(op_convolution_bwd_filters);
    }

    return {};
}

// This follows the style of `build_convolution` in `cudnn_emitter.cpp`.
std::vector<std::tuple<uint32_t, float, size_t, bool>> runtime::gpu::get_algo_options(
        const std::shared_ptr<op::Convolution> node
        )
{
    // Get the backend context from the op annotations
    auto annotation_untyped = node->get_op_annotations();
    auto annotation = dynamic_pointer_cast<runtime::gpu::ConvFwdAnnotations>(annotation_untyped);
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
    // from build_convolution in cudnn_emitter.cpp
    cudnnDataType_t data_type = runtime::gpu::get_cudnn_datatype(dtype);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto descriptors = std::make_shared<runtime::gpu::CUDNNDescriptors>();
    auto& tensor_desc_0 =
        runtime::gpu::tensor_descriptor_from_shape(input_shape, data_type, tensor_format, descriptors);
    auto& tensor_desc_1 =
        tensor_descriptor_from_shape(output_shape, data_type, tensor_format, descriptors);
    auto& filter_desc = runtime::gpu::get_cudnn_filter_descriptor(filter_shape, data_type, tensor_format, descriptors);
    auto& conv_desc = runtime::gpu::get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type, descriptors);


    int num_algos;
    int max_algos = 0;
    CUDNN_SAFE_CALL(
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

    // Construct the return type:
    //
    // Algorithm Enum value
    // Run Time
    // Memory Footprint
    std::vector<std::tuple<uint32_t, float, size_t, bool>> return_vec;
    for (auto res: results)
    {
        // Pass back CUDNN_STATUS_ALLOC_FAILED so we can return an error
#ifdef NGRAPH_ALLOW_NONDETERMINISM
        if (res.status == CUDNN_STATUS_SUCCESS)
#else
        if (res.status == CUDNN_STATUS_SUCCESS && res.determinism == CUDNN_DETERMINISTIC)
#endif
        {
            auto tup = std::tuple<uint32_t, float, size_t, bool>(
                    ngraph::runtime::gpu::to_underlying(res.algo),
                    res.time,
                    res.memory,
                    res.status == CUDNN_STATUS_ALLOC_FAILED
                    );

            return_vec.push_back(tup);
        }
    }

    return return_vec;
}

std::vector<std::tuple<uint32_t, float, size_t, bool>> runtime::gpu::get_algo_options(
        const std::shared_ptr<op::ConvolutionBackpropData> node
        )
{
    // Get the backend context from the op annotations
    auto annotation_untyped = node->get_op_annotations();
    auto annotation = dynamic_pointer_cast<runtime::gpu::ConvBwdDataAnnotations>(annotation_untyped);
    NGRAPH_ASSERT(annotation);
    std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx = annotation->get_context();

    cudnnHandle_t cudnn_handle = *ctx->m_runtime_context->cudnn_handle;

    auto& args = node->get_inputs();
    auto& out = node->get_outputs();

    // For some reason, the order of arguments is filpped for `input_tensor_shape1 and 
    // `input_filter_shape` from cudnn_emitter.cpp ...
    auto input_tensor_shape = args[1].get_shape();
    auto input_filter_shape = args[0].get_shape();
    auto output_tensor_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides_forward();
    Strides window_movement_strides = node->get_window_movement_strides_forward();
    Strides data_dilation_strides = node->get_data_dilation_strides_forward();
    CoordinateDiff padding_below_diff = node->get_padding_below_forward();
    CoordinateDiff padding_above_diff = node->get_padding_above_forward();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    // Padding Schenanigans
    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }

    auto output_shape_padded = output_tensor_shape;
    Shape padding_below_back(output_tensor_shape.size(), 0);
    Shape padding_interior_back(output_tensor_shape.size(), 1);
    size_t i = padding_below_back.size() - padding_below.size();
    size_t j = 0;
    for (; i < padding_below_back.size(); i++)
    {
        padding_below_back[i] = padding_below[j];
        padding_interior_back[i] = data_dilation_strides[j];
        j++;
    }

    Shape padding_interior(data_dilation_strides);

    // from build_convolution_backward_data in cudnn_emitter.cpp
    const cudnnDataType_t data_type = get_cudnn_datatype(output_type);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto descriptors = std::make_shared<runtime::gpu::CUDNNDescriptors>();
    auto& tensor_desc_0 = runtime::gpu::tensor_descriptor_from_shape(
            input_tensor_shape, data_type, tensor_format, descriptors);
    auto& tensor_desc_1 = runtime::gpu::tensor_descriptor_from_shape(
            output_tensor_shape, data_type, tensor_format, descriptors);
    auto& filter_desc = runtime::gpu::get_cudnn_filter_descriptor(
            input_filter_shape, 
            data_type, 
            tensor_format,
            descriptors);
    auto& conv_desc = runtime::gpu::get_cudnn_convolution_descriptor(
        padding_below, window_movement_strides, window_dilation_strides, mode, data_type, descriptors);

    //std::cout << input_tensor_shape << std::endl;
    //std::cout << output_tensor_shape << std::endl;
    //std::cout << input_filter_shape << std::endl;

    int num_algos;
    int max_algos = 0;
    CUDNN_SAFE_CALL(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle, &max_algos));
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> results(max_algos);
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle,
                                         filter_desc,
                                         tensor_desc_0,
                                         conv_desc,
                                         tensor_desc_1,
                                         static_cast<int>(results.size()),
                                         &num_algos,
                                         results.data()));
    results.resize(num_algos);
    // Construct the return type:
    //
    // Algorithm Enum value
    // Run Time
    // Memory Footprint
    std::vector<std::tuple<uint32_t, float, size_t, bool>> return_vec;
    for (auto res: results)
    {
#ifdef NGRAPH_ALLOW_NONDETERMINISM
        if (res.status == CUDNN_STATUS_SUCCESS)
#else
        if (res.status == CUDNN_STATUS_SUCCESS && res.determinism == CUDNN_DETERMINISTIC)
#endif
        {
            auto tup = std::tuple<uint32_t, float, size_t, bool>(
                    ngraph::runtime::gpu::to_underlying(res.algo),
                    res.time,
                    res.memory,
                    res.status == CUDNN_STATUS_ALLOC_FAILED
                    );

            return_vec.push_back(tup);
        }
    }

    return return_vec;
}

std::vector<std::tuple<uint32_t, float, size_t, bool>> runtime::gpu::get_algo_options(
        const std::shared_ptr<op::ConvolutionBackpropFilters> node)
{
    // Get the backend context from the op annotations
    auto annotation_untyped = node->get_op_annotations();
    auto annotation = dynamic_pointer_cast<runtime::gpu::ConvBwdFilterAnnotations>(annotation_untyped);
    NGRAPH_ASSERT(annotation);
    std::shared_ptr<runtime::gpu::GPU_Backend::BackendContext> ctx = annotation->get_context();

    cudnnHandle_t cudnn_handle = *ctx->m_runtime_context->cudnn_handle;
    auto& args = node->get_inputs();
    auto& out = node->get_outputs();
    // Again, these two seem to be flipped here from the `cudnn_emitter.cpp` version of
    // these routines ...
    auto input_tensor_shape_0 = args[0].get_shape();
    auto input_tensor_shape_1 = args[1].get_shape();
    auto output_filter_shape = out[0].get_shape();
    Strides window_dilation_strides = node->get_window_dilation_strides_forward();
    Strides window_movement_strides = node->get_window_movement_strides_forward();
    Strides data_dilation_strides = node->get_data_dilation_strides_forward();
    CoordinateDiff padding_below_diff = node->get_padding_below_forward();
    CoordinateDiff padding_above_diff = node->get_padding_above_forward();
    auto input_type = args[0].get_element_type().c_type_string();
    auto output_type = out[0].get_element_type().c_type_string();

    Shape padding_below(padding_below_diff.size(), 0);
    Shape padding_above(padding_above_diff.size(), 0);
    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
        padding_above[i] = static_cast<size_t>(padding_above_diff[i]);
    }

    //std::cout << input_tensor_shape_0 << std::endl;
    //std::cout << input_tensor_shape_1 << std::endl;
    //std::cout << output_filter_shape << std::endl;

    const cudnnDataType_t data_type = get_cudnn_datatype(output_type);
    const cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    auto descriptors = std::make_shared<runtime::gpu::CUDNNDescriptors>();
    auto& tensor_desc_0 = runtime::gpu::tensor_descriptor_from_shape(
            input_tensor_shape_0, 
            data_type, 
            tensor_format,
            descriptors);
    auto& tensor_desc_1 = runtime::gpu::tensor_descriptor_from_shape(
            input_tensor_shape_1, 
            data_type, 
            tensor_format,
            descriptors);
    auto& filter_desc = runtime::gpu::get_cudnn_filter_descriptor(
            output_filter_shape, 
            data_type, 
            tensor_format,
            descriptors);
    auto& conv_desc = runtime::gpu::get_cudnn_convolution_descriptor(
        padding_below, 
        window_movement_strides, 
        window_dilation_strides, 
        mode, 
        data_type,
        descriptors);

    int num_algos;
    int max_algos = 0;
    CUDNN_SAFE_CALL(
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &max_algos));
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> results(max_algos);
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle,
                                         tensor_desc_0,
                                         tensor_desc_1,
                                         conv_desc,
                                         filter_desc,
                                         static_cast<int>(results.size()),
                                         &num_algos,
                                         results.data()));
    results.resize(num_algos);

    // Construct the return type:
    //
    // Algorithm Enum value
    // Run Time
    // Memory Footprint
    //std::cout << "Filters Max Algos: " << max_algos << std::endl;
    //std::cout << "Filters Num Algos: " << num_algos << std::endl;
    std::vector<std::tuple<uint32_t, float, size_t, bool>> return_vec;
    for (auto res: results)
    {
#ifdef NGRAPH_ALLOW_NONDETERMINISM
        if (res.status == CUDNN_STATUS_SUCCESS)
#else
        if (res.status == CUDNN_STATUS_SUCCESS && res.determinism == CUDNN_DETERMINISTIC)
#endif
        {
            auto tup = std::tuple<uint32_t, float, size_t, bool>(
                    ngraph::runtime::gpu::to_underlying(res.algo),
                    res.time,
                    res.memory,
                    res.status == CUDNN_STATUS_ALLOC_FAILED
                    );

            return_vec.push_back(tup);
        } else {
            //std::cout << "Backward Failed - Status: " << res.status << std::endl;
        }
    }

    return return_vec;
}
