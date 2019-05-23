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

    // This is wrong - need to fix
    return make_shared<Move>(new_args.at(0), m_n);
}

op::Move::Move(const shared_ptr<Node>& input, size_t n)
    : Op("Move", {input})
    , m_n{n}
{
    constructor_validate_and_infer_types();

    set_output_type(0, get_input_element_type(m_n), get_input_partial_shape(m_n));

    // Manually assign layouts since this node is usually inserted after compilation
    auto tv = input->get_output_tensor_ptr(m_n);
    get_output_tensor_ptr(0)->set_tensor_layout(tv->get_tensor_layout());

    // Need to manually assign the correct output form the input
    m_inputs.clear();
    m_inputs.emplace_back(this, 0, input->get_outputs().at(m_n));
}
