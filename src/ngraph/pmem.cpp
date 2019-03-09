#ifdef NGRAPH_PMDK_ENABLE
// stdlib stuff
#include <cstdlib>
#include <string>

// ngraph/pmemobj stuff
#include "pmem.hpp"
#include "libpmemobj.h"
#include "ngraph/log.hpp"

using namespace ngraph;

// Initialize Instance
pmem::PMEMManager* pmem::PMEMManager::m_instance = nullptr;

pmem::PMEMManager* ngraph::pmem::PMEMManager::getinstance()
{
    if (m_instance == nullptr) return m_instance;
    return m_instance = new pmem::PMEMManager();
}

void ngraph::pmem::PMEMManager::open_pool(std::string pool_name)
{
    PMEMobjpool* pmem_pool = pmemobj_open(pool_name.c_str(), "");
    
    m_pool_name = pool_name;
    m_pmem_pool = pmem_pool;
}

void ngraph::pmem::PMEMManager::create_pool(std::string pool_name, size_t size)
{
    PMEMobjpool* pmem_pool = pmemobj_create(pool_name.c_str(), "", size, 0666);

    m_pool_name = pool_name;
    m_pmem_pool = pmem_pool;
}

void ngraph::pmem::PMEMManager::close_pool()
{
    pmemobj_close(m_pmem_pool);
}

void* ngraph::pmem::PMEMManager::pmem_malloc(size_t size)
{
    PMEMManager* manager = pmem::PMEMManager::getinstance();

    PMEMoid oidp;
    auto yolo = pmemobj_zalloc(manager->getpool(), &oidp, size, 0);
    void* ptr = pmemobj_direct(oidp);

    if (yolo == -1)
    {
        NGRAPH_ERR << strerror(errno) << std::endl;
    }
    return ptr;
}

void ngraph::pmem::PMEMManager::pmem_free(void* ptr)
{
    PMEMoid oidp = pmemobj_oid(ptr);
    pmemobj_free(&oidp);
}

#endif
