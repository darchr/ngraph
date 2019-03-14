#pragma once

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            // This is to be used in conjunction with the `set_pool_number` / 
            // `get_pool_number` methods of descriptor::Tensor.
            //
            // Those methods default to a pool number of 0, which we will use for DRAM.
            //
            // A pool number of 1 will then be used for PMEM
            enum class MemoryLocation : std::size_t {
                DRAM = 0,
                PMEM = 1
            };
        }
    }
}
