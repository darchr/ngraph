#ifdef NGRAPH_PMDK_ENABLE
// stdlib stuff
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdio.h>

#include "pmem.hpp"
#include "libpmem.h"
#include "ngraph/util.hpp"
#include "ngraph/log.hpp"

using namespace ngraph;

// Initialize Instance
pmem::PMEMManager* pmem::PMEMManager::m_instance = nullptr;

pmem::PMEMManager* ngraph::pmem::PMEMManager::getinstance()
{
    if (m_instance != nullptr) return m_instance;
    return m_instance = new pmem::PMEMManager();
}

void* ngraph::pmem::PMEMManager::malloc(size_t size)
{
    std::string pool_name = m_pool_dir + "pool_" + std::to_string(m_count);
    m_count++;

    // Make sure we at least make a block sized file.
    size = std::max(size, static_cast<size_t>(4096));

    void* pmem_pool = pmem_map_file(
        pool_name.c_str(), size, PMEM_FILE_CREATE, 0666, nullptr, nullptr
    );

    // Register this pool and size
    m_memory_map[pmem_pool] = std::pair<size_t, std::string>(size, pool_name);
    return pmem_pool;
}

void ngraph::pmem::PMEMManager::free(void* ptr)
{
    auto search = m_memory_map.find(ptr);
    // If we dont' have this pointer, try a normal "free"
    if (search == m_memory_map.end())
    {
        ngraph_free(ptr); 
    } else {
        // Unmap the pool, delete the file, and remove the pointer.
        pmem_unmap(ptr, search->second.first);
        remove(search->second.second.c_str());
        m_memory_map.erase(ptr);
    }
}

bool ngraph::pmem::PMEMManager::is_persistent(void* ptr)
{
    return m_memory_map.find(ptr) != m_memory_map.end();
}

void* ngraph::pmem::pmem_malloc(size_t size)
{
    PMEMManager* manager = pmem::PMEMManager::getinstance();

    void* ptr = manager->malloc(size);
    return ptr;
}

void ngraph::pmem::pmem_free(void* ptr)
{
    PMEMManager* manager = pmem::PMEMManager::getinstance();
    manager->free(ptr);
}

bool ngraph::pmem::is_persistent_ptr(void* ptr)
{
    PMEMManager* manager = pmem::PMEMManager::getinstance();
    return manager->is_persistent(ptr);
}

#endif
