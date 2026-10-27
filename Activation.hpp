#pragma once

#include "MemoryAllocator.hpp"

namespace pankey{

    namespace AI{

        enum class Activation{
            Linear,
            Relu,
            LeakyRelu
        };

        struct ActivationModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = Activation;
            using SIZE_TYPE = int;
        };

        inline float activate(Activation a_activation, float a_value){
            if(a_activation == Activation::Relu){
                return a_value > 0.0f ? a_value : 0.0f;
            }
            if(a_activation == Activation::LeakyRelu){
                return a_value > 0.0f ? a_value : a_value * 0.01f;
            }
            return a_value;
        }

        inline float activationDerivative(Activation a_activation, float a_value){
            if(a_activation == Activation::Relu){
                return a_value > 0.0f ? 1.0f : 0.0f;
            }
            if(a_activation == Activation::LeakyRelu){
                return a_value > 0.0f ? 1.0f : 0.01f;
            }
            return 1.0f;
        }
    }
}