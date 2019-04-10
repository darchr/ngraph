// The inclusion of this file is determined by the CMAKE build process.
#pragma once

#ifdef NGRAPH_PMDK_ENABLE
#include <string>
#include <unordered_map>

#include "libpmem.h"

namespace ngraph
{
    namespace pmem
    {
        class PMEMManager
        {
        public:
            static PMEMManager* getinstance();

            std::string get_pool_dir() { return m_pool_dir; }
            void set_pool_dir(std::string dir) { m_pool_dir = dir; }

            void* malloc(size_t size); 
            void free(void* ptr);
            bool is_persistent(void* ptr);

        private:
            std::string m_pool_dir;

            // Single pool manager for everything - used to control the allocation of
            // persistent memory. Eventually, we should probably move this into the runtime
            // context of the CPU, but for now its simpler to do this.
            static PMEMManager* m_instance;
            int64_t m_count = 0;
            std::unordered_map<void*, std::pair<size_t, std::string>> m_memory_map;
        };

        // Alloc-dealloc
        void* pmem_malloc(size_t size);
        void pmem_free(void*);
        bool is_persistent_ptr(void* ptr);
    }
}
#endif
