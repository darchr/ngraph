#pragma once

#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPU_BACKEND_API HeterogenousMemoryAssignment : 
                     public ngraph::pass::FunctionPass;
                {
                public:
                    virtual bool 
                        run_on_function(std::shared_ptr<ngraph::Function> function) override;
                };
            }
        }
    }
}
