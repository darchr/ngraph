#include "ngraph/runtime/cpu/op/move.hpp"
#include "ngraph/util.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::Move::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Move>(new_args.at(0));
}

op::Move::Move(const shared_ptr<Node>& input)
    : Op("Move", check_single_output_args({input}))
{
    constructor_validate_and_infer_types();

    set_output_type(0, get_input_element_type(0), get_input_shape(0));

    // Manually assign layouts since this node is usually inserted after compilation
    auto tv = input->get_output_tensor_ptr(0);
    auto tvl = tv->get_tensor_layout();

    this->get_output_tensor_ptr(0)->set_tensor_layout(tv->get_tensor_layout());
}
