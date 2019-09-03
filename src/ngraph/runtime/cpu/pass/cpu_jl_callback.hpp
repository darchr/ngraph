#include "ngraph/function.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"

 namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPU_JL_Callback : public ngraph::pass::FunctionPass
                {
                public:
                    CPU_JL_Callback(void (*jl_callback)())
                        : m_jl_callback(jl_callback)
                    {
                    }

                     virtual bool
                        run_on_function(const std::shared_ptr<ngraph::Function> f);

                 private:
                    void (*m_jl_callback)();
                };
            }
        }
    }
}
