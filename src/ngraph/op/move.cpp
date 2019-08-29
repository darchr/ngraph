#include "ngraph/op/move.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

// MARK: Note - these are meant to be inserted after most compiler passes run.
//
// Not save for general use.
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

    // Manually setup out output
    set_output_type(0, this->input(0).get_element_type(), this->input(0).get_partial_shape());

    // Clear this node as a dependency for the other outputs of the source
    int i = 0;
    for (auto& output: input->outputs())
    {
        output.remove_target_input(this->input(i++));
    }

    // Manually assign layouts since this node is usually inserted after compilation
    //auto tv = input->get_output_tensor_ptr(m_n);
    auto tv = input->output(m_n).get_tensor_ptr();
    get_output_tensor_ptr(0)->set_tensor_layout(tv->get_tensor_layout());

    // Need to manually assign the correct output form the input
    nuke_inputs();
    this->input(0).replace_source_output(input->output(m_n)); 
}

shared_ptr<Node> op::MoveAsync::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<MoveAsync>(new_args.at(0), m_n, m_across);
}

op::MoveAsync::MoveAsync(const shared_ptr<Node>& input, size_t n, const shared_ptr<Node>& across)
    : Op("MoveAsync", {input})
    , m_n{n}
    , m_across{across}
{
    constructor_validate_and_infer_types();

    // Manually setup out output
    set_output_type(0, get_input_element_type(m_n), get_input_partial_shape(m_n));

    // Clear this node as a dependency for the other outputs of the source
    int i = 0;
    for (auto& output: input->outputs())
    {
        output.remove_target_input(this->input(i++));
    }

    // Manually assign layouts since this node is usually inserted after compilation
    //auto tv = input->get_output_tensor_ptr(m_n);
    auto tv = input->output(m_n).get_tensor_ptr();
    get_output_tensor_ptr(0)->set_tensor_layout(tv->get_tensor_layout());

    // Need to manually assign the correct output form the input
    nuke_inputs();
    this->input(0).replace_source_output(input->output(m_n)); 
}
