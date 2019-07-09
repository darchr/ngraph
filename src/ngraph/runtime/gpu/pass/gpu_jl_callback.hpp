#include "ngraph/function.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                class GPU_JL_Callback : public ngraph::pass::FunctionPass
                {
                public:
                    GPU_JL_Callback(
                            std::shared_ptr<GPU_Backend::BackendContext> context,
                            void (*jl_callback)(),
                            bool prepare
                            )
                        : m_context(context)
                        , m_jl_callback(jl_callback)
                        , m_prepare(prepare)
                    {
                    }

                    virtual bool
                        run_on_function(const std::shared_ptr<ngraph::Function> f);

                private:
                    bool m_prepare = true;
                    std::shared_ptr<GPU_Backend::BackendContext> m_context;
                    void (*m_jl_callback)();
                };
            }
        }
    }
}
