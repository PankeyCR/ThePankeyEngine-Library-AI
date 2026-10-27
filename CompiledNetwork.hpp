#pragma once

#include "Neuron.hpp"
#include "Optimizer.hpp"

namespace pankey{

    namespace AI{

        struct CompiledNetwork{

            TensorSizes m_layer_sizes;
            TensorSizes m_layer_offsets;
            TensorSizes m_layer_connection_offsets;
            TensorSizes m_sources;
            TensorSizes m_targets;
            TensorValues m_weights;
            TensorValues m_biases;
            pankey::DataStructure::Array::ArrayPointer<ActivationModel> m_activations;

            void clear(){
                m_layer_sizes.clear();
                m_layer_offsets.clear();
                m_layer_connection_offsets.clear();
                m_sources.clear();
                m_targets.clear();
                m_weights.clear();
                m_biases.clear();
                m_activations.clear();
            }

            bool compile(const TensorSizes& a_layer_sizes, const Neurons& a_neurons){
                clear();

                if(a_layer_sizes.isEmpty()){
                    return false;
                }

                int i_neuron_count = 0;
                for(int i = 0; i < a_layer_sizes.length(); ++i){
                    const int i_layer_size = a_layer_sizes.getFast(i);
                    if(i_layer_size <= 0){
                        clear();
                        return false;
                    }
                    m_layer_offsets.add(i_neuron_count);
                    i_neuron_count += i_layer_size;
                }

                if(i_neuron_count != a_neurons.length()){
                    clear();
                    return false;
                }

                m_layer_sizes = a_layer_sizes;

                for(int i = 0; i < a_layer_sizes.length(); ++i){
                    m_layer_connection_offsets.add(m_sources.length());
                    const int i_first_neuron = m_layer_offsets.getFast(i);
                    const int i_last_neuron = i_first_neuron + a_layer_sizes.getFast(i);

                    for(int j = i_first_neuron; j < i_last_neuron; ++j){
                        const Neuron i_neuron = a_neurons.getFast(j);
                        m_biases.add(i_neuron.m_bias);
                        m_activations.add(i_neuron.m_activation);

                        for(int k = 0; k < i_neuron.connectionCount(); ++k){
                            const int i_target = i_neuron.m_targets.getFast(k);
                            if(i_target < 0 || i_target >= i_neuron_count || layerAt(i_target) <= i){
                                clear();
                                return false;
                            }
                            m_sources.add(j);
                            m_targets.add(i_target);
                            m_weights.add(i_neuron.m_weights.getFast(k));
                        }
                    }
                }

                return true;
            }

            bool forward(const Tensor& a_input, Tensor& a_activations) const{
                if(m_layer_sizes.isEmpty() || a_input.rank() != 1 || a_input.size() != m_layer_sizes.getFast(0)){
                    return false;
                }

                TensorSizes i_shape = createTensorSizes(m_biases.length());
                if(!a_activations.create(i_shape)){
                    return false;
                }

                float* i_values = a_activations.m_data.getArrayPointer();
                const float* i_input = a_input.m_data.getArrayPointer();

                for(int i = 0; i < m_biases.length(); ++i){
                    i_values[i] = m_biases.getFast(i);
                }
                for(int i = 0; i < a_input.size(); ++i){
                    i_values[i] = i_input[i];
                }
                for(int i = 0; i < m_layer_sizes.length(); ++i){
                    if(i > 0){
                        activateLayer(i, i_values);
                    }

                    const int i_start = m_layer_connection_offsets.getFast(i);
                    const int i_end = i + 1 < m_layer_connection_offsets.length()
                        ? m_layer_connection_offsets.getFast(i + 1)
                        : m_sources.length();

                    for(int j = i_start; j < i_end; ++j){
                        i_values[m_targets.getFast(j)] += i_values[m_sources.getFast(j)] * m_weights.getFast(j);
                    }
                }

                return true;
            }

            int inputSize() const{
                if(m_layer_sizes.isEmpty()){
                    return 0;
                }
                return m_layer_sizes.getFast(0);
            }

            int outputSize() const{
                if(m_layer_sizes.isEmpty()){
                    return 0;
                }
                return m_layer_sizes.getFast(m_layer_sizes.length() - 1);
            }

            bool output(const Tensor& a_activations, Tensor& a_output) const{
                if(a_activations.rank() != 1 || a_activations.size() != m_biases.length()){
                    return false;
                }

                TensorSizes i_shape = createTensorSizes(outputSize());
                if(!a_output.create(i_shape)){
                    return false;
                }

                const int i_offset = m_layer_offsets.getFast(m_layer_offsets.length() - 1);
                const float* i_values = a_activations.m_data.getArrayPointer();
                float* i_output = a_output.m_data.getArrayPointer();

                for(int i = 0; i < outputSize(); ++i){
                    i_output[i] = i_values[i_offset + i];
                }

                return true;
            }

            bool train(
                const Tensor& a_input,
                const Tensor& a_expected_output,
                float a_learning_rate,
                float& a_loss,
                Optimizer& a_optimizer
            ){
                GradientBuffer i_gradients;
                if(!i_gradients.create(m_weights.length(), m_biases.length())
                    || !accumulateGradient(a_input, a_expected_output, i_gradients, a_loss)){
                    return false;
                }
                return applyGradients(i_gradients, a_learning_rate, 1, a_optimizer);
            }

            bool accumulateGradient(
                const Tensor& a_input,
                const Tensor& a_expected_output,
                GradientBuffer& a_gradients,
                float& a_loss
            ) const{
                if(m_layer_sizes.length() < 2
                    || a_expected_output.rank() != 1
                    || a_expected_output.size() != outputSize()
                    || a_gradients.m_weight_gradients.length() != m_weights.length()
                    || a_gradients.m_bias_gradients.length() != m_biases.length()){
                    return false;
                }

                Tensor i_values;
                if(!forward(a_input, i_values)){
                    return false;
                }
                TensorValues i_deltas;
                i_deltas.createArray(m_biases.length());
                i_deltas.setLength(m_biases.length());

                float* i_delta_values = i_deltas.getArrayPointer();
                const float* i_activation_values = i_values.m_data.getArrayPointer();
                const float* i_expected_values = a_expected_output.m_data.getArrayPointer();
                for(int i = 0; i < i_deltas.length(); ++i){
                    i_delta_values[i] = 0.0f;
                }

                a_loss = 0.0f;
                const int i_output_offset = m_layer_offsets.getFast(m_layer_offsets.length() - 1);
                const float i_loss_scale = 1.0f / static_cast<float>(outputSize());
                for(int i = 0; i < outputSize(); ++i){
                    const int i_neuron_index = i_output_offset + i;
                    const float i_difference = i_activation_values[i_neuron_index] - i_expected_values[i];
                    a_loss += 0.5f * i_difference * i_difference * i_loss_scale;
                    i_delta_values[i_neuron_index] = i_difference * i_loss_scale
                        * activationDerivative(m_activations.getFast(i_neuron_index), i_activation_values[i_neuron_index]);
                }

                for(int i = m_layer_sizes.length() - 2; i > 0; --i){
                    const int i_start = m_layer_connection_offsets.getFast(i);
                    const int i_end = m_layer_connection_offsets.getFast(i + 1);
                    for(int j = i_start; j < i_end; ++j){
                        i_delta_values[m_sources.getFast(j)] += i_delta_values[m_targets.getFast(j)] * m_weights.getFast(j);
                    }
                    applyDerivative(i, i_activation_values, i_delta_values);
                }

                for(int i = 0; i < m_sources.length(); ++i){
                    const float i_gradient = i_activation_values[m_sources.getFast(i)] * i_delta_values[m_targets.getFast(i)];
                    a_gradients.m_weight_gradients.setFast(
                        i,
                        a_gradients.m_weight_gradients.getFast(i) + i_gradient
                    );
                }
                for(int i = m_layer_offsets.getFast(1); i < m_biases.length(); ++i){
                    a_gradients.m_bias_gradients.setFast(
                        i,
                        a_gradients.m_bias_gradients.getFast(i) + i_delta_values[i]
                    );
                }

                return true;
            }

            bool applyGradients(
                const GradientBuffer& a_gradients,
                float a_learning_rate,
                int a_example_count,
                Optimizer& a_optimizer
            ){
                if(m_layer_offsets.length() < 2){
                    return false;
                }
                return a_optimizer.apply(
                    m_weights,
                    m_biases,
                    a_gradients,
                    m_layer_offsets.getFast(1),
                    a_learning_rate,
                    a_example_count
                );
            }

        private:
            void activateLayer(int a_layer, float* a_values) const{
                const int i_first_neuron = m_layer_offsets.getFast(a_layer);
                const int i_last_neuron = i_first_neuron + m_layer_sizes.getFast(a_layer);

                for(int i = i_first_neuron; i < i_last_neuron; ++i){
                    a_values[i] = activate(m_activations.getFast(i), a_values[i]);
                }
            }

            void applyDerivative(int a_layer, const float* a_activations, float* a_deltas) const{
                const int i_first_neuron = m_layer_offsets.getFast(a_layer);
                const int i_last_neuron = i_first_neuron + m_layer_sizes.getFast(a_layer);

                for(int i = i_first_neuron; i < i_last_neuron; ++i){
                    a_deltas[i] *= activationDerivative(m_activations.getFast(i), a_activations[i]);
                }
            }

            int layerAt(int a_neuron_index) const{
                for(int i = m_layer_offsets.length() - 1; i >= 0; --i){
                    if(a_neuron_index >= m_layer_offsets.getFast(i)){
                        return i;
                    }
                }
                return -1;
            }
        };
    }
}