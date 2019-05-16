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

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Lstm : public Op
        {
        public:
            // INPUTS:
            // [0] - {Xt} input tensor of layout TNC, Shape{sequence length*batch_size, feature_size}
            // [1] - recurrent state tensors {ht_1 | ct_1} of Shape{sequence length*batch_size, feature_size}
            // [2] - initializer for the input weights matrix, used for the linear transformation of the inputs.
            // [3] - initializer for the recurrent weights matrix, used for the linear transformation of the recurrent state.
            // [4] - Initializer for the bias vector w.r.to inputs + hidden state (ibh_bias + hbh_bias)

            // OUTPUT VALUE: A tuple with the following structure:
            //   [0] - ht, output tensor with shape (sequence_length*batch_size, num_hidden) .
            //   [1] - {ht | ct} output recurrent state tensor with the same shape as states

            // This version of the LSTM op supports MKLDNN emitter code, this can be used standalone for computing RNN
            // without fusing RNN cell (LSTM)'s across time steps.
            Lstm(std::shared_ptr<Node> src_layer,
                 std::shared_ptr<Node> src_iter,
                 std::shared_ptr<Node> weights_layer,
                 std::shared_ptr<Node> weights_iter,
                 std::shared_ptr<Node> bias,
                 ngraph::runtime::cpu::rnn_utils::rnntype rnn_type);
            Shape get_output_tensor_shape() const { return m_output_tensor_shape; }
            Shape get_output_cell_shape() const { return m_output_cell_shape; }
            ngraph::runtime::cpu::rnn_utils::rnntype get_rnn_type() const { return m_rnntype; }
            size_t get_num_timesteps() const { return m_num_timesteps; }
            size_t get_src_sequence_length() const { return m_src_sequence_length; }
            size_t get_gates_per_cell() const { return m_num_gates_per_cell; }
            size_t get_batch_size() const { return m_batch_size; }
            size_t get_src_layer_feature_size() const { return m_src_layer_feature_size; }
            size_t get_src_iter_feature_size() const { return m_src_iter_feature_size; }
            size_t get_num_cell_states() const { return m_num_cell_states; }
            size_t get_direction() const { return m_direction; }
            size_t get_num_fused_layers() const { return m_num_fused_layers; }
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            Shape m_output_tensor_shape;
            Shape m_output_cell_shape;
            size_t m_num_timesteps;
            size_t m_num_gates_per_cell;
            size_t m_src_sequence_length;
            size_t m_batch_size;
            size_t m_src_layer_feature_size;
            size_t m_src_iter_feature_size;
            size_t m_num_cell_states;
            size_t m_direction;
            size_t m_num_fused_layers;
            ngraph::runtime::cpu::rnn_utils::rnntype m_rnntype;
        };

        class LstmBackprop : public Op
        {
        public:
            LstmBackprop(std::shared_ptr<Node> result_forward,
                         std::shared_ptr<Node> fprop_src_layer,
                         std::shared_ptr<Node> fprop_src_iter,
                         std::shared_ptr<Node> fprop_weights_layer,
                         std::shared_ptr<Node> fprop_weights_iter,
                         std::shared_ptr<Node> fprop_bias,
                         std::shared_ptr<Node> fprop_dst_layer,
                         std::shared_ptr<Node> fprop_dst_iter,
                         std::shared_ptr<Node> diff_dst_layer,
                         std::shared_ptr<Node> diff_dst_iter);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            struct RnnAttributes
            {
                size_t timestep;  /* sequence_len or number of timesteps */
                size_t batch;     /* batch size */
                size_t states;    /* number of recurrent states */
                size_t layer;     /* number of rnn layers */
                size_t direction; /* rnn direction */
                size_t gates;     /* number of gates */
                size_t slc;       /* input feature size */
                size_t sic;       /* hidden state feature size */
            };
            RnnAttributes get_rnn_attributes() const { return m_rnn_attributes; }
            std::shared_ptr<Node> get_fprop_node() const { return m_fprop_node; }
        private:
            std::shared_ptr<Node> m_fprop_node;
            RnnAttributes m_rnn_attributes;
        };
    }
}
