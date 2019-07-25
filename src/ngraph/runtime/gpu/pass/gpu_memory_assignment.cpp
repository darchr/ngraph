#include "ngraph/runtime/gpu/pass/gpu_memory_assignment.hpp"

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/move.hpp"

#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

runtime::gpu::pass::GPUMemoryLayout::GPUMemoryLayout(size_t alignment, bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

bool runtime::gpu::pass::GPUMemoryLayout::run_on_function(shared_ptr<ngraph::Function> function)
{
    // Run the liveness analysis attached to this class
    liveness_analysis(function);

    ngraph::pass::MemoryManager mm(m_alignment, m_disable_memory_sharing);

    // MARK: create a memory allocator for the host CPU.
    //
    // Set alignment to 4096 to match page size.
    ngraph::pass::MemoryManager mm_host(4096, false);

    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
        std::set<const descriptor::Tensor*> reused_inputs;

        if (node->is_op())
        {
            auto op = std::static_pointer_cast<ngraph::op::Op>(node);
            // concat and slice in_place_oi should be treated differently
            if (!std::dynamic_pointer_cast<ngraph::op::Concat>(node) &&
                !std::dynamic_pointer_cast<ngraph::op::Slice>(node))
            {
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        auto output = &node->get_outputs().at(oi_pair.output).get_tensor();
                        auto input = &node->get_inputs().at(oi_pair.input).get_tensor();
                        auto input_node =
                            node->get_inputs().at(oi_pair.input).get_output().get_node();

                        // Skip if input and output tensors are in different pools.
                        if (output->get_pool_number() == 1 || input->get_pool_number() == 1)
                        {
                            continue;
                        }

                        // For destructive kernel, this should be the last use
                        // Non-destructive kernels can pass through if memory sharing is disabled
                        if ((node->liveness_free_list.count(input) != 0 ||
                             std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(node) ||
                             (m_disable_memory_sharing && !oi_pair.destructive &&
                              !input_node->is_parameter() && !input_node->is_constant())) &&
                            node->liveness_new_list.count(output) != 0)

                        {
                            NGRAPH_DEBUG << "Reusing " << input->get_name() << " for "
                                         << output->get_name();
                            in_place_outputs.insert({output, input});
                            reused_inputs.insert(input);
                        }
                    }
                }
            }
        }

        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            //std::cout << "C++ - Working on: " << tensor->get_name() << std::endl;
            size_t offset;
            if (tensor->get_pool_number() == 1)
            {
                offset = mm_host.allocate(tensor->size());
            } else {
                offset = in_place_outputs.count(tensor)
                                    ? in_place_outputs.at(tensor)->get_pool_offset()
                                    : mm.allocate(tensor->size());
            }
            tensor->set_pool_offset(offset);
        }

        if (!m_disable_memory_sharing)
        {
            for (const descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (tensor->get_pool_number() == 1)
                {
                    mm_host.free(tensor->get_pool_offset());
                }
                else if (reused_inputs.count(tensor) == 0)
                {
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
    }
    function->set_temporary_pool_size(mm.max_allocated());
    function->set_pmem_pool_size(mm_host.max_allocated());

    std::cout << "C++ (GPU Memory Assignment) Temp Pool Size: " << mm.max_allocated() << std::endl;
    std::cout << "C++ (GPU Memory Assignment) Host Pool Size: " << mm_host.max_allocated() << std::endl;

    return false;
}

// Liveness analysis
void runtime::gpu::pass::GPUMemoryLayout::liveness_analysis(shared_ptr<Function> function)
{
    list<shared_ptr<Node>> ops = function->get_ordered_ops();

    // Make sure function input, output, and constant tensors aren't counted
    unordered_set<descriptor::Tensor*> persistent_tensors;
    unordered_set<descriptor::Tensor*> output_tensors;
    for (const shared_ptr<op::Parameter>& node : function->get_parameters())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<op::Result>& node : function->get_results())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
            output_tensors.insert(&tensor);
        }
    }
    for (const shared_ptr<Node>& node : ops)
    {
        if (auto constant_node = dynamic_pointer_cast<ngraph::op::Constant>(node))
        {
            for (size_t i = 0; i < constant_node->get_output_size(); ++i)
            {
                descriptor::Tensor& tensor = constant_node->get_output_tensor(i);
                persistent_tensors.insert(&tensor);
            }
        }
    }

    unordered_set<descriptor::Tensor*> currently_live;

    
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        const shared_ptr<Node>& node = *it;
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        unordered_set<descriptor::Tensor*> input_tensor_decls;
        for (descriptor::Input& input_decl : node->get_inputs())
        {
            descriptor::Tensor& tensor = input_decl.get_tensor();
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> output_tensor_decls;
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            if (persistent_tensors.find(&tensor) == persistent_tensors.end())
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> free_tensor_decls;
        unordered_set<descriptor::Tensor*> new_tensor_decls;
        unordered_set<descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (descriptor::Tensor* tensor_decl : all_tensor_decls)
        {
            if (currently_live.find(tensor_decl) == currently_live.end())
            {
                // this is the last node that value is seen in
                // delete it at the end of the op
                currently_live.insert(tensor_decl);
                if (output_tensors.find(tensor_decl) == output_tensors.end())
                {
                    // Don't free output tensors
                    free_tensor_decls.insert(tensor_decl);
                }
            }
        }

        for (descriptor::Tensor* output_decl : output_tensor_decls)
        {
            auto currently_live_it = currently_live.find(output_decl);
            if (currently_live_it != currently_live.end())
            {
                new_tensor_decls.insert(output_decl);
                currently_live.erase(currently_live_it);
            }
        }

        node->liveness_free_list = free_tensor_decls;
        node->liveness_new_list = new_tensor_decls;
    }

    /////
    ///// Deal with workspace assignment
    /////

    // Keep track of the previous node.
    //
    // We need to do this so we can do workspace allocation correctly
    set<Node*> followed_by_async;
    set<Node*> last_move_in_chain;
    shared_ptr<Node> previous_node;
    for (const shared_ptr<Node>& node : ops)
    {
        if (dynamic_pointer_cast<op::MoveAsync>(node) && 
                !dynamic_pointer_cast<op::MoveAsync>(previous_node))
        {
            followed_by_async.insert(previous_node.get());
        } else if (!dynamic_pointer_cast<op::MoveAsync>(node) &&
                    dynamic_pointer_cast<op::MoveAsync>(previous_node))
        {
            last_move_in_chain.insert(previous_node.get());
        }
        previous_node = node;
    }

    vector<descriptor::Tensor*> pending_workspaces;
    for (const shared_ptr<Node>& node : ops)
    {
        // Check if this op has a workspace tensor. If so, add it to the both the new and
        // free list
        if (runtime::gpu::has_algo(node.get()))
        {
            descriptor::Tensor* workspace_tensor = 
                runtime::gpu::get_workspace_tensor(node.get()).get();

            node->liveness_new_list.insert(workspace_tensor);
            node->liveness_free_list.insert(workspace_tensor);
            //if (followed_by_async.count(node.get()) == 0)
            //{
            //} else {
            //    std::cout << "At Node: " << node->get_name() << std::endl;
            //    std::cout << "Inserting Workspace into queue:" << workspace_tensor->get_name() << std::endl;
            //    pending_workspaces.push_back(workspace_tensor);
            //}
        }

        // If this node is followed by a MoveAsync, we must defer the freelist until
        // after the last node in the MoveAsync chain
        if (followed_by_async.count(node.get()) == 1)
        {
            for (descriptor::Tensor* tensor : node->liveness_free_list)
            {
                pending_workspaces.push_back(tensor);
            }
            node->liveness_free_list.clear();
        } 

        // If we're transitioning out of a MoveAsync chain, check to see if there are 
        // pending workspace to free - and free them
        if (last_move_in_chain.count(node.get()) == 1)
        {
            for (descriptor::Tensor* tensor: pending_workspaces)
            {
                std::cout << "At Node: " << node->get_name() << std::endl;
                std::cout << "Freeing Workspace after MoveAsync:" 
                          << tensor->get_name() 
                          << std::endl;

                node->liveness_free_list.insert(tensor);
            }
            pending_workspaces.clear();
        }
    }
}
