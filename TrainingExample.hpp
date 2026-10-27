#pragma once

#include "Tensor.hpp"

namespace pankey{

    namespace AI{

        struct TrainingExample{
            Tensor m_input;
            Tensor m_expected_output;
        };

        struct TrainingExampleModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = TrainingExample;
            using SIZE_TYPE = int;
        };

        using TrainingExamples = pankey::DataStructure::Array::ArrayPointer<TrainingExampleModel>;

        struct GradientBuffer{
            TensorValues m_weight_gradients;
            TensorValues m_bias_gradients;

            bool create(int a_weight_count, int a_bias_count){
                if(a_weight_count < 0 || a_bias_count < 0){
                    return false;
                }

                m_weight_gradients.createArray(a_weight_count);
                m_weight_gradients.setLength(a_weight_count);
                m_bias_gradients.createArray(a_bias_count);
                m_bias_gradients.setLength(a_bias_count);

                for(int i = 0; i < a_weight_count; ++i){
                    m_weight_gradients.setFast(i, 0.0f);
                }
                for(int i = 0; i < a_bias_count; ++i){
                    m_bias_gradients.setFast(i, 0.0f);
                }
                return true;
            }
        };
    }
}