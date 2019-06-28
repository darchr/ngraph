#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                class GPUMemoryLayout : public ngraph::pass::FunctionPass
                {
                public:
                    GPUMemoryLayout(size_t alignment = 1, bool disable_memory_sharing = false);
                    bool run_on_function(std::shared_ptr<ngraph::Function>) override;

                private:
                    // liveness analysis to build new and free list for each node
                    //
                    // need a custom liveness pass to handle Node workspaces
                    void liveness_analysis(std::shared_ptr<ngraph::Function> function);

                    size_t m_alignment;
                    bool m_disable_memory_sharing;
                };
            }
        }
    }
}
