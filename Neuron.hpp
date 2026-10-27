#pragma once

#include "Activation.hpp"
#include "Tensor.hpp"

namespace pankey{

    namespace AI{

        struct Neuron{

            float m_bias = 0.0f;
            Activation m_activation = Activation::Linear;
            TensorSizes m_targets;
            TensorValues m_weights;

            bool connect(int a_target_neuron, float a_weight){
                if(a_target_neuron < 0){
                    return false;
                }

                return m_targets.add(a_target_neuron) && m_weights.add(a_weight);
            }

            int connectionCount() const{
                return m_targets.length();
            }
        };

        struct NeuronModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = Neuron;
            using SIZE_TYPE = int;
        };

        using Neurons = pankey::DataStructure::Array::ArrayPointer<NeuronModel>;
    }
}