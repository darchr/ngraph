#include "ngraph/runtime/gpu/op/sync.hpp"

ngraph::op::SyncBarrier::SyncBarrier(std::shared_ptr<ngraph::Node> arg)
    : Op("SyncBarrier", check_single_output_args({arg}))
{
    constructor_validate_and_infer_types();

    // Determine the shapes and types of the output
    //
    // Swiped this from UnaryElementWise ops
    auto args_et_pshape = validate_and_infer_elementwise_args();
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_ASSERT(this, args_et.is_dynamic() || args_et != element::boolean)
        << "Arguments cannot have boolean element type (argument element type: " << args_et << ").";

    set_output_type(0, args_et, args_pshape);
}

std::shared_ptr<ngraph::Node> ngraph::op::SyncBarrier::copy_with_new_args(
        const ngraph::NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<SyncBarrier>(new_args.at(0));
}
