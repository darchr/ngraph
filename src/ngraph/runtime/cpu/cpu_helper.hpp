// Expose some functionality (mostly MKLDNN related) to my external hooks for helping
// with format conversion and recording.
//
// Note, avoid having mkldnn.hpp in this header file because whenever I try to add 
// MKLDNN as a dependency of the Julia wrapper code, LLVM breaks

#pragma once

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            bool input_needs_conversion(
                const std::shared_ptr<ngraph::Node>& node,
                size_t input_index);

            int64_t get_input_format_int(const std::shared_ptr<ngraph::Node>& node, size_t index);
            std::string get_mkldnn_string(int64_t enum_int);
        }
    }
}
