#include <cstdint>
#include <string>
#include <vector>

#include "cpu_heterogenous_memory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/util.hpp"

// For now, I'll use a JSON file to relay the inputs and outputs to set as heterogenous
// memory, encoded in an environmental variable.
//
// This is horrible, horrible.
#include "nlohmann/json.hpp"

using namespace ngraph;

bool runtime::cpu::pass::HeterogenousMemoryAssignment::run_on_function(
        std::shared_ptr<Function> function)
{

}
