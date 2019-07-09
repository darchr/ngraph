//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/gpu/cudnn_descriptors.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"

#include <cudnn.h>

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            /// \brief Annotations added to graph ops by GPU backend passes
            class GPUOpAnnotations : public ngraph::op::util::OpAnnotations
            {
            public:
                virtual ~GPUOpAnnotations() = default;
            };

            class BatchNormBackpropAnnotations : public GPUOpAnnotations
            {
            public:
                ~BatchNormBackpropAnnotations() = default;
                bool has_inverted_variance() { return m_inv_variance; }
                void set_inverted_variance(bool b) { m_inv_variance = b; }
            private:
                bool m_inv_variance = false;
            };


            // cudnn algorithm selection
            class ConvFwdAnnotations : public GPUOpAnnotations
            {
            public:
                ~ConvFwdAnnotations() = default;

                cudnnConvolutionFwdAlgo_t get_algo() { return m_algo; }
                void set_algo(cudnnConvolutionFwdAlgo_t algo) { m_algo = algo; }
                std::shared_ptr<ngraph::descriptor::Tensor> get_workspace_tensor() { return m_workspace_tensor; }
                void set_workspace_tensor(std::shared_ptr<ngraph::descriptor::Tensor> tensor) { m_workspace_tensor = tensor; }

                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> get_context()
                {
                    return m_context;
                }

                void clear_context() { m_context = nullptr; }
                void set_context(std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> context)
                {
                    m_context = context;
                }
            private:
                cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                std::shared_ptr<ngraph::descriptor::Tensor>  m_workspace_tensor;
                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> m_context;
            };

            // BwdData
            class ConvBwdDataAnnotations : public GPUOpAnnotations
            {
            public:
                ~ConvBwdDataAnnotations() = default;

                cudnnConvolutionBwdDataAlgo_t get_algo() { return m_algo; }
                void set_algo(cudnnConvolutionBwdDataAlgo_t algo) { m_algo = algo; }
                std::shared_ptr<ngraph::descriptor::Tensor> get_workspace_tensor() { return m_workspace_tensor; }
                void set_workspace_tensor(std::shared_ptr<ngraph::descriptor::Tensor> tensor) { m_workspace_tensor = tensor; }

                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> get_context()
                {
                    return m_context;
                }

                void clear_context() { m_context = nullptr; }
                void set_context(std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> context)
                {
                    m_context = context;
                }
            private:
                cudnnConvolutionBwdDataAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
                std::shared_ptr<ngraph::descriptor::Tensor>  m_workspace_tensor;
                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> m_context;
            };

            // BwdFilter
            class ConvBwdFilterAnnotations : public GPUOpAnnotations
            {
            public:
                ~ConvBwdFilterAnnotations() = default;

                cudnnConvolutionBwdFilterAlgo_t get_algo() { return m_algo; }
                void set_algo(cudnnConvolutionBwdFilterAlgo_t algo) { m_algo = algo; }
                std::shared_ptr<ngraph::descriptor::Tensor> get_workspace_tensor() { return m_workspace_tensor; }
                void set_workspace_tensor(std::shared_ptr<ngraph::descriptor::Tensor> tensor) { m_workspace_tensor = tensor; }

                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> get_context()
                {
                    return m_context;
                }

                void clear_context() { m_context = nullptr; }
                void set_context(std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> context)
                {
                    m_context = context;
                }
            private:
                cudnnConvolutionBwdFilterAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
                std::shared_ptr<ngraph::descriptor::Tensor>  m_workspace_tensor;
                std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> m_context;
            };

            // Forwarding types for accessing annotations
            template<typename T> struct FwdAnnotationType;
            template<typename T> struct FwdAlgoType;

            // Forwarding for Convolutions
            template<> struct FwdAnnotationType<ngraph::op::Convolution>
            {
                typedef ConvFwdAnnotations type;
            };

            template<> struct FwdAlgoType<ngraph::op::Convolution>
            {
                typedef cudnnConvolutionFwdAlgo_t type;
            };

            template<> struct FwdAnnotationType<const ngraph::op::Convolution>
            {
                typedef ConvFwdAnnotations type;
            };

            template<> struct FwdAlgoType<const ngraph::op::Convolution>
            {
                typedef cudnnConvolutionFwdAlgo_t type;
            };

            // Forwarding for backward convolutions
            template<> struct FwdAnnotationType<ngraph::op::ConvolutionBackpropData>
            {
                typedef ConvBwdDataAnnotations type;
            };

            template<> struct FwdAlgoType<ngraph::op::ConvolutionBackpropData>
            {
                typedef cudnnConvolutionBwdDataAlgo_t type;
            };

            template<> struct FwdAnnotationType<const ngraph::op::ConvolutionBackpropData>
            {
                typedef ConvBwdDataAnnotations type;
            };

            template<> struct FwdAlgoType<const ngraph::op::ConvolutionBackpropData>
            {
                typedef cudnnConvolutionBwdDataAlgo_t type;
            };

            // Forwarding for backward convolutions filter
            template<> struct FwdAnnotationType<ngraph::op::ConvolutionBackpropFilters>
            {
                typedef ConvBwdFilterAnnotations type;
            };

            template<> struct FwdAlgoType<ngraph::op::ConvolutionBackpropFilters>
            {
                typedef cudnnConvolutionBwdFilterAlgo_t type;
            };

            template<> struct FwdAnnotationType<const ngraph::op::ConvolutionBackpropFilters>
            {
                typedef ConvBwdFilterAnnotations type;
            };

            template<> struct FwdAlgoType<const ngraph::op::ConvolutionBackpropFilters>
            {
                typedef cudnnConvolutionBwdFilterAlgo_t type;
            };

            /////
            ///// Query Functions
            /////

            // TODO: Put these in a cpp file once I figure out how exactly to do that
            template<typename T>
            inline bool has_algo(T* node)
            {
                auto annotations = node->get_op_annotations();
                if (annotations)
                {
                    auto cast_anno = std::dynamic_pointer_cast<typename FwdAnnotationType<T>::type>(annotations);
                    if (cast_anno)
                    {
                        return true;
                    }
                }
                return false;
            }

            // Query functions to make this less annoying
            template<>
            inline bool has_algo(Node* node)
            {
                // Forward Convolution
                auto op_convolution = dynamic_cast<ngraph::op::Convolution*>(node);
                if (op_convolution)
                {
                    return has_algo(op_convolution);
                }

                // Backward Data Convolution
                auto op_bwd_data_convolution = dynamic_cast<ngraph::op::ConvolutionBackpropData*>(node);
                if (op_bwd_data_convolution)
                {
                    return has_algo(op_bwd_data_convolution);
                }

                // Backward Filter Convolution
                auto op_bwd_filter_convolution = dynamic_cast<ngraph::op::ConvolutionBackpropFilters*>(node);
                if (op_bwd_filter_convolution)
                {
                    return has_algo(op_bwd_filter_convolution);
                }
                return false;
            }

            // Annotation Initializer
            template<typename T>
            inline bool do_annotation(
                    T* node,
                    std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> ctx
                )
            {
                auto annotation = std::make_shared<typename FwdAnnotationType<T>::type>();
                annotation->set_context(ctx);
                node->set_op_annotations(annotation);
                return true;
            }

            template<>
            inline bool do_annotation(
                    Node* node,
                    std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> ctx
                )
            {
                auto op_convolution = dynamic_cast<ngraph::op::Convolution*>(node);
                if (op_convolution)
                {
                    return do_annotation(op_convolution, ctx);
                }
                auto op_convolution_bwd_data = dynamic_cast<ngraph::op::ConvolutionBackpropData*>(node);
                if (op_convolution_bwd_data)
                {
                    return do_annotation(op_convolution_bwd_data, ctx);
                }
                auto op_convolution_bwd_filters = dynamic_cast<ngraph::op::ConvolutionBackpropFilters*>(node);
                if (op_convolution_bwd_filters)
                {
                    return do_annotation(op_convolution_bwd_filters, ctx);
                }
                return false;
            }

            // Get the algorithm type from an annotated node
            template<typename T>
            inline typename FwdAlgoType<T>::type get_algo(T* node)
            {
                // Make sure this actually has an algorithm assigned
                NGRAPH_ASSERT(has_algo(node));
                auto annotations = node->get_op_annotations();
                auto cast_anno = std::dynamic_pointer_cast<typename FwdAnnotationType<T>::type>(annotations);
                return cast_anno->get_algo();
            }

            // Get the workspace tensor for the node
            template<typename T>
            inline std::shared_ptr<ngraph::descriptor::Tensor> get_workspace_tensor(T* node)
            {
                //NGRAPH_ASSERT(has_algo(node));
                auto annotations = node->get_op_annotations();
                auto cast_anno = std::dynamic_pointer_cast<typename FwdAnnotationType<T>::type>(annotations);
                return cast_anno->get_workspace_tensor();
            }

            // Entry point for getting workspace tensors
            template<>
            inline std::shared_ptr<ngraph::descriptor::Tensor> get_workspace_tensor(Node* node)
            {
                auto op_convolution = dynamic_cast<ngraph::op::Convolution*>(node);
                if (op_convolution)
                {
                    return get_workspace_tensor(op_convolution);
                }
                auto op_convolution_bwd_data = dynamic_cast<ngraph::op::ConvolutionBackpropData*>(node);
                if (op_convolution_bwd_data)
                {
                    return get_workspace_tensor(op_convolution_bwd_data);
                }
                auto op_convolution_bwd_filters = dynamic_cast<ngraph::op::ConvolutionBackpropFilters*>(node);
                if (op_convolution_bwd_filters)
                {
                    return get_workspace_tensor(op_convolution_bwd_filters);
                }
                // If everything fails, return a nullptr
                return {};
            }

            // CUDNN stuff because it's reallh hard to get to the cudnn stuff inside
            // cudnn_emitter.hpp
            cudnnTensorDescriptor_t&
                tensor_descriptor_from_shape(const Shape& shape,
                                             const cudnnDataType_t data_type,
                                             const cudnnTensorFormat_t tensor_format, 
                                             const std::shared_ptr<runtime::gpu::CUDNNDescriptors> descriptors);


            cudnnDataType_t get_cudnn_datatype(std::string dtype);
            cudnnDataType_t get_cudnn_datatype(const element::Type& dtype);

            cudnnFilterDescriptor_t&
                get_cudnn_filter_descriptor(const Shape& shape,
                                            const cudnnDataType_t data_type,
                                            const cudnnTensorFormat_t tensor_format,
                                            const std::shared_ptr<runtime::gpu::CUDNNDescriptors> descriptors);

            cudnnConvolutionDescriptor_t&
                get_cudnn_convolution_descriptor(const Shape& padding,
                                                 const Strides& window_movement_strides,
                                                 const Strides& window_dilation_strides,
                                                 cudnnConvolutionMode_t mode,
                                                 cudnnDataType_t data_type,
                                                 const std::shared_ptr<runtime::gpu::CUDNNDescriptors> descriptors);
        }
    }
}

