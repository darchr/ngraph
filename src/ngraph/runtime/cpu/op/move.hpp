#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace op
    {
        class Move : public Op
        {
        public:
            CPU_BACKEND_API Move(const std::shared_ptr<Node>& input, size_t n);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_n() const { return m_n; }

            // TODO: Maybe implement this
            //
            // For now, insertion of MOVE nodes happens after backpropogation runs,
            // so we don't have to worry yet about implementing this function.
            //
            //virtual void generate_adjoints(autodiff::Adjoints& adjoints,
            //                               const NodeVector& deltas) override;
        protected:
            size_t m_n;
        };

        class MoveAsync : public Op
        {
        public:
            CPU_BACKEND_API MoveAsync(
                    const std::shared_ptr<Node>& input, 
                    size_t n,
                    const std::string across
            );

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_n() const { return m_n; }

            const std::string& get_fellow() { return m_across_name; }

            // TODO: Maybe implement this
            //
            // For now, insertion of MOVE nodes happens after backpropogation runs,
            // so we don't have to worry yet about implementing this function.
            //
            //virtual void generate_adjoints(autodiff::Adjoints& adjoints,
            //                               const NodeVector& deltas) override;
        protected:
            size_t m_n;
            // The node to move in parallel with
            const std::string m_across_name;
        };
    }
}
