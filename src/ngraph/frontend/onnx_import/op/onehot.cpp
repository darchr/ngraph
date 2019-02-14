//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <memory>

#include "exceptions.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "onehot.hpp"
#include "utils/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector onehot(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    auto indices =
                        std::make_shared<ngraph::op::Convert>(inputs.at(0), element::i64);
                    auto indices_shape = indices->get_shape();
                    auto depth = inputs.at(1);
                    auto values = inputs.at(2);
                    std::shared_ptr<ngraph::Node> off_value = std::make_shared<ngraph::op::Convert>(
                        std::make_shared<ngraph::op::Slice>(values, Coordinate{0}, Coordinate{1}),
                        element::i64);
                    std::shared_ptr<ngraph::Node> on_value = std::make_shared<ngraph::op::Convert>(
                        std::make_shared<ngraph::op::Slice>(values, Coordinate{1}, Coordinate{2}),
                        element::i64);
                    auto axis = node.get_attribute_value<int64_t>("axis", -1);

                    if (axis < 0)
                    {
                        axis += indices_shape.size() + 1;
                    }

                    ASSERT_VALID_ARGUMENT(node, (axis >= 0) && (axis <= indices_shape.size()))
                        << "provided 'axis' attribute is not valid.";

                    auto cdepth = std::dynamic_pointer_cast<ngraph::op::Constant>(depth);

                    ASSERT_VALID_ARGUMENT(node, cdepth) << "Only static depth is supported";

                    std::int64_t depth_value = cdepth->get_vector<std::int64_t>()[0];
                    auto output_shape = indices_shape;
                    output_shape.insert(std::next(std::begin(output_shape), axis), depth_value);

                    std::shared_ptr<ngraph::Node> one_hot =
                        std::make_shared<ngraph::op::OneHot>(indices, output_shape, axis);
                    on_value = numpy_style_broadcast_for_binary_operation(one_hot, on_value)[1];
                    off_value = numpy_style_broadcast_for_binary_operation(one_hot, off_value)[1];
                    one_hot = one_hot * (on_value - off_value) + off_value;
                    return {std::make_shared<ngraph::op::Convert>(
                        one_hot, inputs.at(0)->get_element_type())};
                }

            } // namespace set_1

        } //namespace op

    } // namespace  onnx_import

} // namespace  ngraph
