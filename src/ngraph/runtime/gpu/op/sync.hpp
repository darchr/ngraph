#pragma once

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class SyncBarrier : public ngraph::op::Op
        {
        public:
            // We rely on control dependencies to get this into the function
            SyncBarrier(std::shared_ptr<ngraph::Node> arg);

        protected:
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        };
    }
}
