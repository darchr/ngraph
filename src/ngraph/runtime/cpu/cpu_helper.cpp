#include <typeindex>
#include <typeinfo>

#include <mkldnn.hpp>
#include "mkldnn_utils.hpp"

#include "cpu_helper.hpp"

using namespace std;
using namespace ngraph;

bool runtime::cpu::input_needs_conversion(const shared_ptr<Node>& node, size_t input_index)
{
    // Check to see if a this input has a defined memory format. If so, return `true`.
    const mkldnn::memory::desc& mkldnn_md = 
        runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node.get(), input_index);

    return mkldnn_md.data.format != mkldnn::memory::format::format_undef;
}

