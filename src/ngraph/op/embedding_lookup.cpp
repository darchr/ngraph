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

#include "ngraph/log.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

const string op::EmbeddingLookup::type_name{"EmbeddingLookup"};

void op::EmbeddingLookup::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(1);

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          arg1_shape.rank().is_dynamic() ||
                              static_cast<size_t>(arg1_shape.rank()) == 2,
                          "weights are expected to be a matrix");

    PartialShape result_shape;
    if (arg0_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(static_cast<size_t>(arg0_shape.rank()) + 1);
        for (size_t i = 0; i < static_cast<size_t>(arg0_shape.rank()); i++)
        {
            result_dims[i] = arg0_shape[i];
        }

        result_dims[result_dims.size() - 1] =
            arg1_shape.rank().is_static() ? arg1_shape[1] : Dimension::dynamic();
        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::EmbeddingLookup::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<EmbeddingLookup>(new_args.at(0), new_args.at(1));
}

//shared_ptr<op::Reshape> make_reshape_axes_to_front(const Output<Node>& n,
//                                                   const Shape& front_shape,
//                                                   const Shape& back_shape)
//{
//    AxisVector input_order;
//    Shape output_shape;
//
//    for (size_t i = 0; i < back_shape.size(); i++)
//    {
//        input_order.push_back(front_shape.size() + i);
//        output_shape.push_back(back_shape[i]);
//    }
//
//    for (size_t i = 0; i < front_shape.size(); i++)
//    {
//        input_order.push_back(i);
//        output_shape.push_back(front_shape[i]);
//    }
//
//    return make_shared<op::Reshape>(n, input_order, output_shape);
//}

void op::EmbeddingLookup::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto data = input_value(0);
    auto weights = input_value(1);

    // The hack we use here is to transpose the indices and expand them using the "one_hot"
    // operator.
    //const PartialShape output_shape = weights.get_shape();
    Shape output_shape; 
    for (auto i: data.get_shape())
    {
        output_shape.push_back(i);
    }
    output_shape.push_back(weights.get_shape().at(0));

    auto data_expand_transpose = make_shared<OneHot>(data, output_shape, output_shape.size() - 1);
    auto converted = make_shared<Convert>(data_expand_transpose, weights.get_element_type());
    
    // Now - we need to transpose the converted one-hot vecto
    Shape I_shape; 
    Shape J_shape;
    Shape K_shape;

    auto x_shape = converted->get_shape();
    auto y_shape = weights.get_shape();

    size_t reduction_axes_count = 1;
    I_shape.insert(I_shape.begin(), x_shape.begin(), x_shape.end() - reduction_axes_count);
    J_shape.insert(J_shape.begin(), y_shape.begin(), y_shape.begin() + reduction_axes_count);
    K_shape.insert(K_shape.begin(), y_shape.begin() + J_shape.size(), y_shape.end());

    auto x_reshaped = make_reshape_axes_to_front(converted, I_shape, J_shape);               // JI
    auto x_reshaped_dot_delta = make_shared<Dot>(x_reshaped, delta, I_shape.size()); // JI.IK->JK
    adjoints.add_delta(weights, x_reshaped_dot_delta);
    
    //adjoints.add_delta(weights, make_shared<op::Dot>(converted, delta));

    //auto x_shape = x.get_shape();          // shape IJ
    //auto y_shape = y.get_shape();          // shape JK
    //auto delta_shape = delta->get_shape(); // shape IK

    //Shape I_shape;
    //Shape J_shape;
    //Shape K_shape;
    //I_shape.insert(I_shape.begin(), x_shape.begin(), x_shape.end() - m_reduction_axes_count);
    //J_shape.insert(J_shape.begin(), y_shape.begin(), y_shape.begin() + m_reduction_axes_count);
    //K_shape.insert(K_shape.begin(), y_shape.begin() + J_shape.size(), y_shape.end());

    //auto y_reshaped = make_reshape_axes_to_front(y, J_shape, K_shape);               // KJ
    //auto delta_dot_y_reshaped = make_shared<Dot>(delta, y_reshaped, K_shape.size()); // IK.KJ->IJ
    //adjoints.add_delta(x, delta_dot_y_reshaped);

    //auto x_reshaped = make_reshape_axes_to_front(x, I_shape, J_shape);               // JI
    //auto x_reshaped_dot_delta = make_shared<Dot>(x_reshaped, delta, I_shape.size()); // JI.IK->JK
    //adjoints.add_delta(y, x_reshaped_dot_delta);
}
