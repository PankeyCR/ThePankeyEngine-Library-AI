#pragma once

#include <cmath>
#include <cstdint>

#include "Brain.hpp"
#include "IntegerArray.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            using pankey::Type::Array::IntegerArray;

            struct ExecutionPlan{
                IntegerArray m_inputs;
                IntegerArray m_outputs;
                IntegerArray m_execution_order;
                bool m_is_dense = false;
                bool m_is_valid = true;
            };

            struct DenseExecutionPlan : public ExecutionPlan{
                IntegerArray m_layer_sizes;
                IntegerArray m_layer_offsets;
                IntegerArray m_brain_indices;

                DenseExecutionPlan(){
                    this->m_is_dense = true;
                    this->m_is_valid = true;
                }
            };

            struct DynamicExecutionPlan : public ExecutionPlan{
                IntegerArray m_node_brain_indices;
                IntegerArray m_node_layer_indices;
                IntegerArray m_connection_offsets;
                IntegerArray m_connection_targets;
                IntegerArray m_connection_weight_indices;

                DynamicExecutionPlan(){
                    this->m_is_dense = false;
                    this->m_is_valid = true;
                }
            };

            struct DenseNetwork{
                IntegerArray m_index;
                IntegerArray m_topology;
                IntegerArray m_layer_offsets;
                IntegerArray m_brain_indices;
                IntegerArray m_input_indices;
                IntegerArray m_output_indices;
                bool m_kind = false;

                DenseNetwork() = default;

                DenseNetwork(const IntegerArray& a_index,
                             const IntegerArray& a_topology,
                             bool a_kind = false)
                    : m_index(a_index),
                      m_topology(a_topology),
                      m_kind(a_kind){
                        compileMetadata();
                    }

                void compileMetadata(){
                    if(this->m_topology.isEmpty()){
                        return;
                    }

                    const int i_layer_count = this->m_topology.length();
                    int i_total = 0;
                    this->m_layer_offsets.createArray(i_layer_count + 1);
                    this->m_layer_offsets.setLength(i_layer_count + 1);
                    for(int i = 0; i < i_layer_count; ++i){
                        const int i_layer_size = this->m_topology.getFast(i);
                        this->m_layer_offsets.setFast(i, i_total);
                        i_total += i_layer_size;
                    }
                    this->m_layer_offsets.setFast(i_layer_count, i_total);

                    this->m_brain_indices.createArray(i_total);
                    this->m_brain_indices.setLength(i_total);
                    this->m_input_indices.createArray(i_total);
                    this->m_input_indices.setLength(i_total);
                    this->m_output_indices.createArray(i_total);
                    this->m_output_indices.setLength(i_total);

                    for(int i = 0; i < i_total; ++i){
                        const int i_index = (this->m_index.length() > i) ? this->m_index.getFast(i) : i;
                        this->m_brain_indices.setFast(i, i_index);
                        this->m_input_indices.setFast(i, i);
                        this->m_output_indices.setFast(i, i);
                    }
                }

                DenseExecutionPlan compile() const{
                    DenseExecutionPlan i_plan;
                    if(this->m_topology.isEmpty()){
                        i_plan.m_is_valid = false;
                        return i_plan;
                    }

                    const int i_layer_count = this->m_topology.length();
                    int i_total = 0;
                    i_plan.m_layer_sizes = this->m_topology;
                    i_plan.m_layer_offsets.createArray(i_layer_count + 1);
                    i_plan.m_layer_offsets.setLength(i_layer_count + 1);
                    for(int i = 0; i < i_layer_count; ++i){
                        const int i_layer_size = this->m_topology.getFast(i);
                        i_plan.m_layer_offsets.setFast(i, i_total);
                        i_total += i_layer_size;
                    }
                    i_plan.m_layer_offsets.setFast(i_layer_count, i_total);

                    i_plan.m_brain_indices.createArray(i_total);
                    i_plan.m_brain_indices.setLength(i_total);
                    i_plan.m_inputs.createArray(i_layer_count > 0 ? this->m_topology.getFast(0) : 0);
                    i_plan.m_inputs.setLength(i_layer_count > 0 ? this->m_topology.getFast(0) : 0);
                    i_plan.m_outputs.createArray(i_layer_count > 0 ? this->m_topology.getFast(i_layer_count - 1) : 0);
                    i_plan.m_outputs.setLength(i_layer_count > 0 ? this->m_topology.getFast(i_layer_count - 1) : 0);
                    i_plan.m_execution_order.createArray(i_total);
                    i_plan.m_execution_order.setLength(i_total);

                    for(int i = 0; i < i_total; ++i){
                        const int i_index = (this->m_index.length() > i) ? this->m_index.getFast(i) : i;
                        i_plan.m_brain_indices.setFast(i, i_index);
                        i_plan.m_execution_order.setFast(i, i);
                    }

                    const int i_input_count = (i_layer_count > 0) ? this->m_topology.getFast(0) : 0;
                    const int i_output_count = (i_layer_count > 0) ? this->m_topology.getFast(i_layer_count - 1) : 0;
                    for(int i = 0; i < i_input_count; ++i){
                        i_plan.m_inputs.setFast(i, i);
                    }
                    for(int i = 0; i < i_output_count; ++i){
                        i_plan.m_outputs.setFast(i, i_total - i_output_count + i);
                    }

                    i_plan.m_is_valid = true;
                    return i_plan;
                }
            };

            struct DynamicNetwork{
                IntegerArray m_node_brain_indices;
                IntegerArray m_node_layer_indices;
                IntegerArray m_connection_offsets;
                IntegerArray m_connection_targets;
                IntegerArray m_connection_weight_indices;
                IntegerArray m_input_nodes;
                IntegerArray m_output_nodes;

                DynamicNetwork() = default;

                DynamicNetwork(const IntegerArray& a_node_brain_indices,
                               const IntegerArray& a_node_layer_indices,
                               const IntegerArray& a_connection_offsets,
                               const IntegerArray& a_connection_targets,
                               const IntegerArray& a_connection_weight_indices,
                               const IntegerArray& a_input_nodes = IntegerArray(),
                               const IntegerArray& a_output_nodes = IntegerArray())
                    : m_node_brain_indices(a_node_brain_indices),
                      m_node_layer_indices(a_node_layer_indices),
                      m_connection_offsets(a_connection_offsets),
                      m_connection_targets(a_connection_targets),
                      m_connection_weight_indices(a_connection_weight_indices),
                      m_input_nodes(a_input_nodes),
                      m_output_nodes(a_output_nodes){}

                DynamicExecutionPlan compile() const{
                    DynamicExecutionPlan i_plan;
                    i_plan.m_node_brain_indices = this->m_node_brain_indices;
                    i_plan.m_node_layer_indices = this->m_node_layer_indices;
                    i_plan.m_connection_offsets = this->m_connection_offsets;
                    i_plan.m_connection_targets = this->m_connection_targets;
                    i_plan.m_connection_weight_indices = this->m_connection_weight_indices;
                    i_plan.m_inputs = this->m_input_nodes;
                    i_plan.m_outputs = this->m_output_nodes;
                    i_plan.m_execution_order.createArray(this->m_node_brain_indices.length());
                    i_plan.m_execution_order.setLength(this->m_node_brain_indices.length());
                    for(int i = 0; i < this->m_node_brain_indices.length(); ++i){
                        i_plan.m_execution_order.setFast(i, i);
                    }
                    i_plan.m_is_valid = !this->m_node_brain_indices.isEmpty();
                    return i_plan;
                }
            };

            class Runtime{
                public:
                    Runtime() = default;

                    Runtime(Brain<>& a_brain, const DenseExecutionPlan& a_plan)
                        : m_brain(&a_brain), m_dense_plan(a_plan), m_uses_dense(true){}

                    Runtime(Brain<>& a_brain, const DynamicExecutionPlan& a_plan)
                        : m_brain(&a_brain), m_dynamic_plan(a_plan), m_uses_dense(false){}

                    Runtime(Brain<>& a_brain, const DenseNetwork& a_network)
                        : m_brain(&a_brain), m_dense_plan(a_network.compile()), m_uses_dense(true){}

                    Runtime(Brain<>& a_brain, const DynamicNetwork& a_network)
                        : m_brain(&a_brain), m_dynamic_plan(a_network.compile()), m_uses_dense(false){}

                    void execute(){
                        if(!m_brain){
                            return;
                        }
                        if(m_uses_dense){
                            executeDense();
                            return;
                        }
                        executeDynamic();
                    }

                    void executeLayer(const int a_layer){
                        if(!m_brain || !m_uses_dense){
                            return;
                        }
                        const int i_layer_count = m_dense_plan.m_layer_sizes.length();
                        if(a_layer < 0 || a_layer >= i_layer_count){
                            return;
                        }
                        const int i_prev_size = (a_layer > 0) ? m_dense_plan.m_layer_sizes.getFast(a_layer - 1) : 0;
                        const int i_current_size = m_dense_plan.m_layer_sizes.getFast(a_layer);
                        const int i_prev_offset = (a_layer > 0) ? m_dense_plan.m_layer_offsets.getFast(a_layer - 1) : 0;
                        const int i_current_offset = m_dense_plan.m_layer_offsets.getFast(a_layer);

                        for(int i_output = 0; i_output < i_current_size; ++i_output){
                            const int i_output_index = i_current_offset + i_output;
                            const int i_brain_output = m_dense_plan.m_brain_indices.getFast(i_output_index);
                            float i_sum = m_brain->m_biases.getFast(i_brain_output);
                            for(int i_input = 0; i_input < i_prev_size; ++i_input){
                                const int i_input_index = i_prev_offset + i_input;
                                const int i_brain_input = m_dense_plan.m_brain_indices.getFast(i_input_index);
                                const int i_weight_index = (i_output * i_prev_size) + i_input;
                                i_sum += m_brain->m_activations.getFast(i_brain_input) * m_brain->m_weights.getFast(i_weight_index);
                            }
                            m_brain->m_activations.setFast(i_brain_output, i_sum);
                            m_brain->m_activation_deltas.setFast(i_brain_output, i_sum);
                        }
                    }

                    void executeLayers(const int a_start_layer, const int a_end_layer){
                        if(!m_brain || !m_uses_dense){
                            return;
                        }
                        const int i_layer_count = m_dense_plan.m_layer_sizes.length();
                        const int i_start = std::max(1, a_start_layer);
                        const int i_end = std::min(i_layer_count - 1, a_end_layer);
                        for(int i = i_start; i <= i_end; ++i){
                            executeLayer(i);
                        }
                    }

                    void executeNodes(const IntegerArray& a_node_indices){
                        if(!m_brain || m_uses_dense){
                            return;
                        }
                        for(int i = 0; i < a_node_indices.length(); ++i){
                            const int i_node = a_node_indices.getFast(i);
                            if(i_node < 0 || i_node >= m_dynamic_plan.m_node_brain_indices.length()){
                                continue;
                            }
                            const int i_brain_index = m_dynamic_plan.m_node_brain_indices.getFast(i_node);
                            float i_sum = m_brain->m_biases.getFast(i_brain_index);
                            const int i_begin = m_dynamic_plan.m_connection_offsets.getFast(i_node);
                            const int i_end = m_dynamic_plan.m_connection_offsets.getFast(i_node + 1);
                            for(int j = i_begin; j < i_end; ++j){
                                const int i_source = m_dynamic_plan.m_connection_targets.getFast(j);
                                const int i_weight_index = m_dynamic_plan.m_connection_weight_indices.getFast(j);
                                const int i_source_brain = m_dynamic_plan.m_node_brain_indices.getFast(i_source);
                                i_sum += m_brain->m_activations.getFast(i_source_brain) * m_brain->m_weights.getFast(i_weight_index);
                            }
                            m_brain->m_activations.setFast(i_brain_index, i_sum);
                            m_brain->m_activation_deltas.setFast(i_brain_index, i_sum);
                        }
                    }

                private:
                    void executeDense(){
                        if(!m_brain || m_dense_plan.m_layer_sizes.isEmpty()){
                            return;
                        }
                        const int i_layer_count = m_dense_plan.m_layer_sizes.length();
                        for(int i_layer = 1; i_layer < i_layer_count; ++i_layer){
                            executeLayer(i_layer);
                        }
                    }

                    void executeDynamic(){
                        if(!m_brain || m_dynamic_plan.m_node_brain_indices.isEmpty()){
                            return;
                        }
                        for(int i = 0; i < m_dynamic_plan.m_execution_order.length(); ++i){
                            const int i_node = m_dynamic_plan.m_execution_order.getFast(i);
                            const int i_brain_index = m_dynamic_plan.m_node_brain_indices.getFast(i_node);
                            float i_sum = m_brain->m_biases.getFast(i_brain_index);
                            const int i_begin = m_dynamic_plan.m_connection_offsets.getFast(i_node);
                            const int i_end = m_dynamic_plan.m_connection_offsets.getFast(i_node + 1);
                            for(int j = i_begin; j < i_end; ++j){
                                const int i_source = m_dynamic_plan.m_connection_targets.getFast(j);
                                const int i_weight_index = m_dynamic_plan.m_connection_weight_indices.getFast(j);
                                const int i_source_brain = m_dynamic_plan.m_node_brain_indices.getFast(i_source);
                                i_sum += m_brain->m_activations.getFast(i_source_brain) * m_brain->m_weights.getFast(i_weight_index);
                            }
                            m_brain->m_activations.setFast(i_brain_index, i_sum);
                            m_brain->m_activation_deltas.setFast(i_brain_index, i_sum);
                        }
                    }

                    Brain<>* m_brain = nullptr;
                    DenseExecutionPlan m_dense_plan;
                    DynamicExecutionPlan m_dynamic_plan;
                    bool m_uses_dense = true;
            };

            using NeuralNetwork = DenseNetwork;

        }

    }

}
