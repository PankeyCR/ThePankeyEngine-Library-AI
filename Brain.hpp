#pragma once

#include <cstdint>

#include "ArrayPointer.hpp"
#include "FloatArray.hpp"
#include "ArrayPointer_IntModel.hpp"
#include "NeuralNetwork.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            struct BrainPolicy{
                using StrongType = pankey::Type::Array::FloatArrayModel;
                using LightType = pankey::Type::Array::ArrayPointer_IntModel;
            };

            template<class Policy = BrainPolicy>
            struct Brain{
                using StrongArray = pankey::DataStructure::Array::ArrayPointer<typename Policy::StrongType>;
                using LightArray = pankey::DataStructure::Array::ArrayPointer<typename Policy::LightType>;

                StrongArray m_weights;
                LightArray m_light_weights;
                StrongArray m_biases;
                LightArray m_light_biases;
                StrongArray m_activations;
                LightArray m_light_activations;
                StrongArray m_activation_deltas;
                LightArray m_light_activation_deltas;

                void clear(){
                    m_weights.clear();
                    m_light_weights.clear();
                    m_biases.clear();
                    m_light_biases.clear();
                    m_activations.clear();
                    m_light_activations.clear();
                    m_activation_deltas.clear();
                    m_light_activation_deltas.clear();
                }

                void create(uint32_t a_nodeCount,
                            uint32_t a_lightNodeCount,
                            uint32_t a_weightCount,
                            uint32_t a_lightWeightCount){
                    clear();
                    createArray(m_biases, a_nodeCount);
                    createArray(m_activations, a_nodeCount);
                    createArray(m_activation_deltas, a_nodeCount);
                    createArray(m_light_biases, a_lightNodeCount);
                    createArray(m_light_activations, a_lightNodeCount);
                    createArray(m_light_activation_deltas, a_lightNodeCount);
                    createArray(m_weights, a_weightCount);
                    createArray(m_light_weights, a_lightWeightCount);
                    clearArray(m_biases);
                    clearArray(m_activations);
                    clearArray(m_activation_deltas);
                    clearArray(m_light_biases);
                    clearArray(m_light_activations);
                    clearArray(m_light_activation_deltas);
                    clearArray(m_weights);
                    clearArray(m_light_weights);
                }

                DenseNetwork buildNetwork(const IntegerArray& a_index,
                                          const IntegerArray& a_topology,
                                          bool a_kind) const{
                    return DenseNetwork(a_index, a_topology, a_kind);
                }

                DynamicNetwork buildNetwork(const IntegerArrayList& a_index,
                                            const IntegerArray& a_topology,
                                            const IntegerArrayList& a_kind,
                                            const IntegerArrayList& a_connections) const{
                    return DynamicNetwork(a_index, a_topology, a_kind, a_connections);
                }

            private:
                template<class Array>
                static void createArray(Array& a_array, uint32_t a_count){
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
