#pragma once

#include <string>

// This function produces a stack backtrace with demangled function & method names.
namespace ngraph
{
    std::string Backtrace(int skip = 1);
}
