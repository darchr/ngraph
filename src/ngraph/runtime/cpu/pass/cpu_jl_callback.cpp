#include "runtime/cpu/pass/cpu_jl_callback.hpp"

#include "ngraph/function.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/node.hpp"

bool ngraph::runtime::cpu::pass::CPU_JL_Callback::run_on_function(
        std::shared_ptr<ngraph::Function> f)
{
    // Invoke the JL callback!!
    (m_jl_callback)();

    return false;
}
