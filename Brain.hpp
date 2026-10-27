#pragma once

#include "ArrayPointer.hpp"
#include "PrimitiveOwnerArrayRawMap.hpp"

namespace pankey{

    namespace Library{

        namespace AI{
            
            struct Brain{

                pankey::Type::Array::PrimitiveOwnerArrayRawPointerMap<pankey::Utility::Test::CharPointer,void> m_light_weights;

                StrongArray m_weights;
                StrongArray m_biases;
                StrongArray m_activations;
                StrongArray m_activation_deltas;

                void clear(){
                    m_weights.clear();
                    m_biases.clear();
                    m_activations.clear();
                    m_activation_deltas.clear();
                }

                void create(Strong_Size a_nodeCount,
                            Light_Size a_lightNodeCount,
                            Strong_Size a_weightCount,
                            Light_Size a_lightWeightCount){
                    clear();
                    createStrongArray(m_biases, a_nodeCount);
                    createStrongArray(m_activations, a_nodeCount);
                    createStrongArray(m_activation_deltas, a_nodeCount);
                    createLightArray(m_light_biases, a_lightNodeCount);
                    createLightArray(m_light_activations, a_lightNodeCount);
                    createLightArray(m_light_activation_deltas, a_lightNodeCount);
                    createStrongArray(m_weights, a_weightCount);
                    createLightArray(m_light_weights, a_lightWeightCount);
                    clearArray(m_biases);
                    clearArray(m_activations);
                    clearArray(m_activation_deltas);
                    clearArray(m_light_biases);
                    clearArray(m_light_activations);
                    clearArray(m_light_activation_deltas);
                    clearArray(m_weights);
                    clearArray(m_light_weights);
                }

            private:
                template<class Array>
                static void createStrongArray(Array& a_array, Strong_Size a_count){
                    using SizeType = typename Array::SIZE_TYPE;
                    a_array.createArray(static_cast<SizeType>(a_count));
                    a_array.setLength(static_cast<SizeType>(a_count));
                }

                template<class Array>
                static void createLightArray(Array& a_array, Light_Size a_count){
                    using SizeType = typename Array::SIZE_TYPE;
                    a_array.createArray(static_cast<SizeType>(a_count));
                    a_array.setLength(static_cast<SizeType>(a_count));
                }

                template<class Array>
                static void clearArray(Array& a_array){
                    for(typename Array::SIZE_TYPE i = 0; i < a_array.length(); ++i){
                        a_array.setFast(i, 0);
                    }
                }
            };

        }

    }

}
