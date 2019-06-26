#include "runtime/gpu/pass/gpu_jl_callback.hpp"

#include "ngraph/function.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"
//#include "ngraph/gpu_helper.hpp"

bool ngraph::runtime::gpu::pass::GPU_JL_Callback::run_on_function(
        std::shared_ptr<ngraph::Function> f)
{
    // On each of the nodes in the graph, attach op annotations if needed
    for (auto node : f->get_ordered_ops())
    {
        if (has_algo(node.get()))
        {
            do_annotation(node.get(), m_context);
        }
    } 

    // Invoke the JL callback!!
    (m_jl_callback)();

    // TODO: Clean up the references to the backend context
    // for (auto node : f->get_ordered_ops())
    // {
    //     if (has_algo(node.get()))
    //     {
    //         do_annotation(node.get(), m_context);
    //     }
    // }

    return false;
}
