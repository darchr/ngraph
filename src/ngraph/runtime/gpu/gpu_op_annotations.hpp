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

#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/convolution.hpp"
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
            private:
                cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                std::shared_ptr<ngraph::descriptor::Tensor>  m_workspace_tensor;
            };

            // class ConvBwdDataAnnotations : public GPUOpAnnotations
            // {
            // public:
            //     ~ConvBwdDataAnnotations() = default;

            //     bool is_configured() { return m_is_configured; }
            //     cudnnConvolutionBwdDataAlgo_t get_algo() { return m_algo; }
            //     void set_algo(cudnnConvolutionBwdDataAlgo_t algo) { m_algo = algo; }
            //     size_t get_workspace_offset() { return m_workspace_offset; }
            //     void set_workspace_offset(size_t workspace_offset) { m_workspace_offset = workspace_offset; }
            // private:
            //     bool m_is_configured = false;
            //     cudnnConvolutionBwdDataAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            //     size_t m_workspace_offset = -1;
            // };

            // class ConvBwdFilterAnnotations : public GPUOpAnnotations
            // {
            // public:
            //     ~ConvBwdFilterAnnotations() = default;

            //     bool is_configured() { return m_is_configured; }
            //     cudnnConvolutionBwdFilterAlgo_t get_algo() { return m_algo; }
            //     void set_algo(cudnnConvolutionBwdFilterAlgo_t algo) { m_algo = algo; }
            //     size_t get_workspace_offset() { return m_workspace_offset; }
            //     void set_workspace_offset(size_t workspace_offset) { m_workspace_offset = workspace_offset; }
            // private:
            //     bool m_is_configured = false;
            //     cudnnConvolutionBwdFilterAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            //     size_t m_workspace_offset = -1;
            // };

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

            // TODO: Bool function that returns if a op has algorithm selection
            // TODO: profile routine to return algorithm performance and workspace sizes
            //      for the different algorithms
            // TODO: Assign the algorithm and workspace size for the op.
            bool can_select_algo(const std::shared_ptr<Node> node);

            void set_algo(const std::shared_ptr<Node> node, size_t algo_enum, size_t workspace_size);
            void set_algo(const std::shared_ptr<op::Convolution> node, size_t algo_enum, size_t workspace_size);

            std::vector<std::tuple<size_t, float, size_t>> get_algo_options(
                    const std::shared_ptr<Node> node,
                    const std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> ctx
                    );

            std::vector<std::tuple<size_t, float, size_t>> get_algo_options(
                    const std::shared_ptr<op::Convolution> node,
                    const std::shared_ptr<ngraph::runtime::gpu::GPU_Backend::BackendContext> ctx
                    );

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
                auto op_convolution = dynamic_cast<ngraph::op::Convolution*>(node);
                if (op_convolution)
                {
                    return has_algo(op_convolution);
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
                // If everything fails, return a nullptr
                return {};
            }

            // CUDNN stuff because it's reallh hard to get to the cudnn stuff inside
            // cudnn_emitter.hpp
            cudnnTensorDescriptor_t&
                tensor_descriptor_from_shape(const Shape& shape,
                                             const cudnnDataType_t data_type,
                                             const cudnnTensorFormat_t tensor_format);

            cudnnDataType_t get_cudnn_datatype(std::string dtype);
            cudnnDataType_t get_cudnn_datatype(const element::Type& dtype);

            cudnnFilterDescriptor_t&
                get_cudnn_filter_descriptor(const Shape& shape,
                                            const cudnnDataType_t data_type,
                                            const cudnnTensorFormat_t tensor_format);

            cudnnConvolutionDescriptor_t&
                get_cudnn_convolution_descriptor(const Shape& padding,
                                                 const Strides& window_movement_strides,
                                                 const Strides& window_dilation_strides,
                                                 cudnnConvolutionMode_t mode,
                                                 cudnnDataType_t data_type);
        }
    }
}

