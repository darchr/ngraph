// The inclusion of this file is determined by the CMAKE build process.
#pragma once

#ifdef NGRAPH_PMDK_ENABLE
#include "libpmemobj.h"
namespace ngraph
{
    namespace pmem
    {
        class PMEMManager
        {
        public:
            static PMEMManager* getinstance();

            PMEMobjpool* getpool() { return m_pmem_pool; }

            // Pool management functions
            void open_pool(std::string pool_name);
            void create_pool(std::string pool_name, size_t size);
            void close_pool();

            // Alloc-dealloc
            void* pmem_malloc(size_t size);
            void pmem_free(void*);
            bool is_persistent_ptr(void* ptr);
            
        private:
            std::string m_pool_name;     
            PMEMobjpool* m_pmem_pool;

            // Single pool manager for everything - used to control the allocation of 
            // persistent memory. Eventually, we should probably move this into the runtime
            // context of the CPU, but for now its simpler to do this.
            static PMEMManager* m_instance;
        };
    }
}
#endif
