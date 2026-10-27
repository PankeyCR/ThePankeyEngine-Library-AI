#pragma once

#include "ArrayPointer_IntModel.hpp"
#include "ArrayPointer_FloatModel.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            struct BrainPolicy{
                using Strong_Type = pankey::Type::Array::ArrayPointer_FloatModel;
                using Light_Type = pankey::Type::Array::ArrayPointer_IntModel;
            };

        }

    }

}
