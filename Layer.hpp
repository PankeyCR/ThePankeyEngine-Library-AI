#pragma once

#include "Tensor.hpp"

namespace pankey{

    namespace AI{

        struct Layer{

            virtual bool forward(const Tensor& a_input, Tensor& a_output) const = 0;

            virtual ~Layer() = default;
        };
    }
}