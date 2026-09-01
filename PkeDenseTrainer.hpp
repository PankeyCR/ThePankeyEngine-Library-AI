#pragma once

#include <cstdint>
#include <vector>

#include "Adam.hpp"
#include "Brain.hpp"
#include "FloatArray.hpp"
#include "PkeConfig.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            class PkeDenseTrainer{
                public:
                    void configure(const PkeConfig& a_config){
                        m_config = a_config;
                        m_optimizer = Adam(m_config.learningRate);
                        m_parameterCount = 0;
                    }

                    bool train(Brain<>& a_brain,
                               const DenseNetwork& a_network,
                               const pankey::Type::Array::FloatArray& a_input,
                               const pankey::Type::Array::FloatArray& a_target){
                        if(a_network.m_topology.length() != 2
                            || a_network.m_kind
                            || a_input.length() != a_network.m_topology.getFast(0)
                            || a_target.length() != a_network.m_topology.getFast(1)){
                            return false;
                        }

                        const int i_inputCount = a_input.length();
                        const int i_outputCount = a_target.length();
                        if(i_inputCount * i_outputCount > a_brain.m_weights.length()
                            || i_outputCount > a_brain.m_biases.length()){
                            return false;
                        }

                        const size_t i_parameterCount = static_cast<size_t>(i_outputCount)
                            + static_cast<size_t>(i_inputCount * i_outputCount);
                        if(m_parameterCount != i_parameterCount){
                            m_optimizer.reset(i_parameterCount);
                            m_parameterCount = i_parameterCount;
                        }
                        std::vector<float> i_parameters(i_parameterCount, 0.0f);
                        for(int i_output = 0; i_output < i_outputCount; ++i_output){
                            i_parameters[i_output] = a_brain.m_biases.getFast(i_output);
                        }
                        for(int i_weight = 0; i_weight < i_inputCount * i_outputCount; ++i_weight){
                            i_parameters[i_outputCount + i_weight] = a_brain.m_weights.getFast(i_weight);
                        }

                        for(uint32_t i_step = 0; i_step < m_config.steps; ++i_step){
                            std::vector<float> i_gradients(i_parameterCount, 0.0f);
                            for(int i_output = 0; i_output < i_outputCount; ++i_output){
                                float i_prediction = i_parameters[i_output];
                                for(int i_input = 0; i_input < i_inputCount; ++i_input){
                                    const int i_weight = i_output * i_inputCount + i_input;
                                    i_prediction += a_input.getFast(i_input) * i_parameters[i_outputCount + i_weight];
                                }
                                const float i_error = i_prediction - a_target.getFast(i_output);
                                i_gradients[i_output] += i_error;
                                for(int i_input = 0; i_input < i_inputCount; ++i_input){
                                    const int i_weight = i_output * i_inputCount + i_input;
                                    i_gradients[i_outputCount + i_weight] += i_error * a_input.getFast(i_input);
                                }
                            }
                            if(!m_optimizer.update(i_parameters, i_gradients)){
                                return false;
                            }
                        }
                        for(int i_output = 0; i_output < i_outputCount; ++i_output){
                            a_brain.m_biases.setFast(i_output, i_parameters[i_output]);
                        }
                        for(int i_weight = 0; i_weight < i_inputCount * i_outputCount; ++i_weight){
                            a_brain.m_weights.setFast(i_weight, i_parameters[i_outputCount + i_weight]);
                        }
                        return true;
                    }

                    float loss(const Brain<>& a_brain,
                               const DenseNetwork& a_network,
                               const pankey::Type::Array::FloatArray& a_input,
                               const pankey::Type::Array::FloatArray& a_target) const{
                        if(a_network.m_topology.length() != 2
                            || a_network.m_kind
                            || a_input.length() != a_network.m_topology.getFast(0)
                            || a_target.length() != a_network.m_topology.getFast(1)){
                            return -1.0f;
                        }

                        const int i_inputCount = a_input.length();
                        const int i_outputCount = a_target.length();
                        float i_loss = 0.0f;
                        for(int i_output = 0; i_output < i_outputCount; ++i_output){
                            float i_prediction = a_brain.m_biases.getFast(i_output);
                            for(int i_input = 0; i_input < i_inputCount; ++i_input){
                                i_prediction += a_input.getFast(i_input)
                                    * a_brain.m_weights.getFast(i_output * i_inputCount + i_input);
                            }
                            const float i_error = i_prediction - a_target.getFast(i_output);
                            i_loss += i_error * i_error;
                        }
                        return i_loss;
                    }

                private:
                    PkeConfig m_config;
                    Adam m_optimizer;
                    size_t m_parameterCount = 0;
            };

        }

    }

}
