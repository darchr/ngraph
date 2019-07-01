#include <type_traits>

#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"


#pragma once

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {

            // TODO: Bool function that returns if a op has algorithm selection
            // TODO: profile routine to return algorithm performance and workspace sizes
            //      for the different algorithms
            // TODO: Assign the algorithm and workspace size for the op.
            bool can_select_algo(const std::shared_ptr<Node> node);

            void set_algo(const std::shared_ptr<Node> node, size_t algo_enum, size_t workspace_size);
            void set_algo(const std::shared_ptr<op::Convolution> node, size_t algo_enum, size_t workspace_size);

            std::vector<std::tuple<uint32_t, float, size_t>> get_algo_options(
                    const std::shared_ptr<Node> node
                    );

            std::vector<std::tuple<uint32_t, float, size_t>> get_algo_options(
                    const std::shared_ptr<op::Convolution> node
                    );

            // Method to convert an ENUM to its umnderlying type.
            template <typename E>
            constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept {
                return static_cast<typename std::underlying_type<E>::type>(e);
            }
        }
    }
}
