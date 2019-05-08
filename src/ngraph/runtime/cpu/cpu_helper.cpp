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

int64_t runtime::cpu::get_input_format_int(const shared_ptr<Node>&node, size_t index)
{
    const mkldnn::memory::desc& mkldnn_md = 
        runtime::cpu::mkldnn_utils::get_input_mkldnn_md(node.get(), index);

    // In MKLDN, the memory formats are set by an enum. Here, we just convert that enum
    // to an integer and return that. Unique integers represent different format kinds,
    // so this will help is differentiate.
    return static_cast<int64_t>(mkldnn_md.data.format);
}

string runtime::cpu::get_mkldnn_string(int64_t enum_int)
{
    // See mkldnn.hpp
    mkldnn::memory::format fmt = static_cast<mkldnn::memory::format>(enum_int);
    return runtime::cpu::mkldnn_utils::get_mkldnn_format_string(fmt);
}
