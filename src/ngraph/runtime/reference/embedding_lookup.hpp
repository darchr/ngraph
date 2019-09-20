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

#include <cmath>
#include <cstring>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void embedding(const U* indices,
                           const T* weights,
                           T* out,
                           size_t indices_count,
                           const Shape& out_shape)
            {
                size_t vec_len = out_shape.at(1);
                T* out_iter = out;
                for (size_t i = 0; i < indices_count; i++)
                {
                    memcpy(out_iter,
                           &weights[vec_len * static_cast<size_t>(indices[i])],
                           sizeof(T) * vec_len);
                    out_iter += vec_len;
                }
            }

            template <typename T, typename U>
            void embedding_backprop(const U* indices, 
                                    const T* deltas,
                                    T* out,
                                    size_t indices_count,
                                    const Shape& out_shape)
            {
                // Zero out the output
                #pragma omp parallel for
                for (size_t i = 0; i < shape_size(out_shape); i++)
                {
                    out[i] = 0;
                }

                // Indices and deltas should have the same size except for the last 
                // dimension.
                //
                // We will iterate over both to perform the update.
                size_t vec_len = out_shape.at(1);
                T* out_iter;

                for (size_t i = 0; i < indices_count; i++)
                {
                    // Get the position in the look up table.
                    out_iter = out + vec_len * indices[i];

                    // Update the vectors
                    for (size_t j = 0; j < vec_len; j++)  
                    {
                        out_iter[j] += deltas[vec_len * i + j];
                    }
                }
            }

        }
    }
}
