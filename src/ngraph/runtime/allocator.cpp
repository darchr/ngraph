//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/runtime/allocator.hpp"
#include "ngraph/util.hpp"
#include "ngraph/pmem.hpp"

ngraph::runtime::Allocator::~Allocator()
{
}

class ngraph::runtime::DefaultAllocator : public ngraph::runtime::Allocator
{
public:
    void* malloc(size_t size, size_t alignment, size_t pool)
    {
        // If allocation succeeds, returns a pointer to the lowest (first) byte in the
        // allocated memory block that is suitably aligned for any scalar type.
        // TODO(pruthvi): replace std::malloc with custom aligned_alloc implementation
        // which is portable and work on all alignment requirement.
        void* ptr = ngraph::ngraph_malloc(size);

        // check for exception
        if (!ptr)
        {
            throw ngraph::ngraph_error("malloc failed to allocate memory of size " +
                                       std::to_string(size));
        }
        return ptr;
    }

    void free(void* ptr)
    {
        if (ptr)
        {
            ngraph::ngraph_free(ptr);
        }
    }
};

ngraph::runtime::Allocator* ngraph::runtime::get_default_allocator()
{
    static DefaultAllocator* allocator = new DefaultAllocator();
    return allocator;
}

class ngraph::runtime::PMMAllocator : public ngraph::runtime::Allocator
{
public:
    void* malloc(size_t size, size_t alignment, size_t pool)
    {
        void* ptr;
        if (pool == 0)
        {
            //std::cout << "PMMAllocator - normal malloc" << std::endl;
            ptr = ngraph::ngraph_malloc(size);
        } else {
            //std::cout << "PMMAllocator - pmm malloc" << std::endl;
            ptr = ngraph::pmem::pmem_malloc(size);
        }

        // check for exception
        if (!ptr)
        {
            throw ngraph::ngraph_error("malloc failed to allocate memory of size " +
                                       std::to_string(size));
        }
        ptr_to_pool[ptr] = pool;
        return ptr;
    }

    void free(void* ptr)
    {
        if (ptr)
        {
            size_t pool = ptr_to_pool.at(ptr);
            // Check what pool to free from.
            if (pool == 0)
            {
                ngraph::ngraph_free(ptr);
            } else {
                ngraph::pmem::pmem_free(ptr);
            }
        }
    }

private:     
    std::map<void*, size_t> ptr_to_pool;
};

ngraph::runtime::Allocator* ngraph::runtime::get_pmm_allocator()
{
    static PMMAllocator* allocator = new PMMAllocator();
    return allocator;
}
