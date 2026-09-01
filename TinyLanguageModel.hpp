#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "Adam.hpp"
#include "Brain.hpp"
#include "PkeLlmConfig.hpp"
#include "TokenSequence.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            class TinyLanguageModel{
                public:
                    TinyLanguageModel(Brain<>& a_brain, const PkeLlmConfig& a_config)
                        : m_brain(a_brain), m_config(a_config), m_optimizer(a_config.learningRate){
                        if(m_config.vocabularySize == 0 || m_config.embeddingSize == 0){
                            m_valid = false;
                            return;
                        }
                        const size_t i_weightCount = static_cast<size_t>(2) * m_config.vocabularySize
                            * m_config.embeddingSize;
                        m_brain.create(m_config.vocabularySize, 0, static_cast<uint32_t>(i_weightCount), 0);
                        m_optimizer.reset(i_weightCount + m_config.vocabularySize);
                        initialize();
                    }

                    bool valid() const{
                        return m_valid;
                    }

                    float loss(const std::vector<TokenSequence>& a_batch) const{
                        if(!m_valid || a_batch.empty()){
                            return -1.0f;
                        }
                        double i_loss = 0.0;
                        uint64_t i_count = 0;
                        for(const TokenSequence& i_sequence : a_batch){
                            if(i_sequence.size() < 2){
                                continue;
                            }
                            for(uint32_t i = 0; i + 1 < i_sequence.size(); ++i){
                                const std::vector<float> i_logits = logits(i_sequence.get(i));
                                const float i_maximum = *std::max_element(i_logits.begin(), i_logits.end());
                                double i_sum = 0.0;
                                for(float i_value : i_logits){
                                    i_sum += std::exp(static_cast<double>(i_value - i_maximum));
                                }
                                const uint32_t i_target = i_sequence.get(i + 1) % m_config.vocabularySize;
                                i_loss += -static_cast<double>(i_logits[i_target] - i_maximum)
                                    + std::log(i_sum);
                                ++i_count;
                            }
                        }
                        return i_count == 0
                            ? -1.0f
                            : static_cast<float>(i_loss / static_cast<double>(i_count));
                    }

                    bool trainBatch(const std::vector<TokenSequence>& a_batch){
                        if(!m_valid || a_batch.empty()){
                            return false;
                        }
                        const size_t i_embeddingSize = static_cast<size_t>(m_config.vocabularySize)
                            * m_config.embeddingSize;
                        const size_t i_outputOffset = i_embeddingSize;
                        const size_t i_biasOffset = i_outputOffset + i_embeddingSize;
                        const size_t i_parameterCount = i_biasOffset + m_config.vocabularySize;
                        std::vector<float> i_parameters(i_parameterCount, 0.0f);
                        std::vector<float> i_gradients(i_parameterCount, 0.0f);
                        for(size_t i = 0; i < i_embeddingSize * 2; ++i){
                            i_parameters[i] = m_brain.m_weights.getFast(static_cast<int>(i));
                        }
                        for(uint32_t i = 0; i < m_config.vocabularySize; ++i){
                            i_parameters[i_biasOffset + i] = m_brain.m_biases.getFast(static_cast<int>(i));
                        }

                        uint64_t i_count = 0;
                        for(const TokenSequence& i_sequence : a_batch){
                            for(uint32_t i = 0; i + 1 < i_sequence.size(); ++i){
                                const uint32_t i_input = i_sequence.get(i) % m_config.vocabularySize;
                                const uint32_t i_target = i_sequence.get(i + 1) % m_config.vocabularySize;
                                std::vector<float> i_logits(m_config.vocabularySize, 0.0f);
                                for(uint32_t i_token = 0; i_token < m_config.vocabularySize; ++i_token){
                                    float i_value = i_parameters[i_biasOffset + i_token];
                                    for(uint32_t i_dimension = 0; i_dimension < m_config.embeddingSize; ++i_dimension){
                                        i_value += i_parameters[i_input * m_config.embeddingSize + i_dimension]
                                            * i_parameters[i_outputOffset + i_dimension * m_config.vocabularySize + i_token];
                                    }
                                    i_logits[i_token] = i_value;
                                }
                                const float i_maximum = *std::max_element(i_logits.begin(), i_logits.end());
                                float i_sum = 0.0f;
                                for(float& i_value : i_logits){
                                    i_value = std::exp(i_value - i_maximum);
                                    i_sum += i_value;
                                }
                                if(!std::isfinite(i_sum) || i_sum <= 0.0f){
                                    return false;
                                }
                                for(float& i_value : i_logits){
                                    i_value /= i_sum;
                                }
                                for(uint32_t i_token = 0; i_token < m_config.vocabularySize; ++i_token){
                                    const float i_error = i_logits[i_token] - (i_token == i_target ? 1.0f : 0.0f);
                                    i_gradients[i_biasOffset + i_token] += i_error;
                                    for(uint32_t i_dimension = 0; i_dimension < m_config.embeddingSize; ++i_dimension){
                                        const size_t i_outputIndex = i_outputOffset
                                            + i_dimension * m_config.vocabularySize + i_token;
                                        i_gradients[i_outputIndex] += i_error
                                            * i_parameters[i_input * m_config.embeddingSize + i_dimension];
                                        i_gradients[i_input * m_config.embeddingSize + i_dimension]
                                            += i_error * i_parameters[i_outputIndex];
                                    }
                                }
                                ++i_count;
                            }
                        }
                        if(i_count == 0){
                            return false;
                        }
                        const float i_scale = 1.0f / static_cast<float>(i_count);
                        for(float& i_value : i_gradients){
                            i_value *= i_scale;
                        }
                        if(!m_optimizer.update(i_parameters, i_gradients)){
                            return false;
                        }
                        for(size_t i = 0; i < i_embeddingSize * 2; ++i){
                            m_brain.m_weights.setFast(static_cast<int>(i), i_parameters[i]);
                        }
                        for(uint32_t i = 0; i < m_config.vocabularySize; ++i){
                            m_brain.m_biases.setFast(static_cast<int>(i), i_parameters[i_biasOffset + i]);
                        }
                        return true;
                    }

                    std::vector<float> logits(uint32_t a_token) const{
                        std::vector<float> i_logits(m_config.vocabularySize, 0.0f);
                        if(!m_valid){
                            return i_logits;
                        }
                        const uint32_t i_input = a_token % m_config.vocabularySize;
                        const size_t i_outputOffset = static_cast<size_t>(m_config.vocabularySize)
                            * m_config.embeddingSize;
                        for(uint32_t i_token = 0; i_token < m_config.vocabularySize; ++i_token){
                            float i_value = m_brain.m_biases.getFast(static_cast<int>(i_token));
                            for(uint32_t i_dimension = 0; i_dimension < m_config.embeddingSize; ++i_dimension){
                                i_value += m_brain.m_weights.getFast(static_cast<int>(i_input * m_config.embeddingSize + i_dimension))
                                    * m_brain.m_weights.getFast(static_cast<int>(i_outputOffset
                                        + i_dimension * m_config.vocabularySize + i_token));
                            }
                            i_logits[i_token] = i_value;
                        }
                        return i_logits;
                    }

                    bool save(const char* a_path) const{
                        std::ofstream i_file(a_path, std::ios::binary);
                        if(!i_file || !m_valid){
                            return false;
                        }
                        const uint32_t i_magic = 0x544C4D31u;
                        const uint32_t i_version = 1;
                        i_file.write(reinterpret_cast<const char*>(&i_magic), sizeof(i_magic));
                        i_file.write(reinterpret_cast<const char*>(&i_version), sizeof(i_version));
                        i_file.write(reinterpret_cast<const char*>(&m_config), sizeof(m_config));
                        const uint32_t i_weightCount = static_cast<uint32_t>(m_brain.m_weights.length());
                        const uint32_t i_biasCount = static_cast<uint32_t>(m_brain.m_biases.length());
                        i_file.write(reinterpret_cast<const char*>(&i_weightCount), sizeof(i_weightCount));
                        i_file.write(reinterpret_cast<const char*>(&i_biasCount), sizeof(i_biasCount));
                        i_file.write(reinterpret_cast<const char*>(m_brain.m_weights.getArrayPointer()),
                                     static_cast<std::streamsize>(i_weightCount * sizeof(float)));
                        i_file.write(reinterpret_cast<const char*>(m_brain.m_biases.getArrayPointer()),
                                     static_cast<std::streamsize>(i_biasCount * sizeof(float)));
                        return static_cast<bool>(i_file);
                    }

                    bool load(const char* a_path){
                        std::ifstream i_file(a_path, std::ios::binary);
                        if(!i_file){
                            return false;
                        }
                        uint32_t i_magic = 0;
                        uint32_t i_version = 0;
                        PkeLlmConfig i_config;
                        uint32_t i_weightCount = 0;
                        uint32_t i_biasCount = 0;
                        i_file.read(reinterpret_cast<char*>(&i_magic), sizeof(i_magic));
                        i_file.read(reinterpret_cast<char*>(&i_version), sizeof(i_version));
                        i_file.read(reinterpret_cast<char*>(&i_config), sizeof(i_config));
                        i_file.read(reinterpret_cast<char*>(&i_weightCount), sizeof(i_weightCount));
                        i_file.read(reinterpret_cast<char*>(&i_biasCount), sizeof(i_biasCount));
                        if(!i_file || i_magic != 0x544C4D31u || i_version != 1
                            || i_config.vocabularySize != m_config.vocabularySize
                            || i_config.embeddingSize != m_config.embeddingSize
                            || i_weightCount != static_cast<uint32_t>(m_brain.m_weights.length())
                            || i_biasCount != static_cast<uint32_t>(m_brain.m_biases.length())){
                            return false;
                        }
                        i_file.read(reinterpret_cast<char*>(m_brain.m_weights.getArrayPointer()),
                                    static_cast<std::streamsize>(i_weightCount * sizeof(float)));
                        i_file.read(reinterpret_cast<char*>(m_brain.m_biases.getArrayPointer()),
                                    static_cast<std::streamsize>(i_biasCount * sizeof(float)));
                        return static_cast<bool>(i_file);
                    }

                private:
                    Brain<>& m_brain;
                    PkeLlmConfig m_config;
                    Adam m_optimizer;
                    bool m_valid = true;

                    void initialize(){
                        std::mt19937 i_generator(m_config.seed);
                        std::normal_distribution<float> i_distribution(0.0f, 0.02f);
                        for(int i = 0; i < m_brain.m_weights.length(); ++i){
                            m_brain.m_weights.setFast(i, i_distribution(i_generator));
                        }
                        for(int i = 0; i < m_brain.m_biases.length(); ++i){
                            m_brain.m_biases.setFast(i, 0.0f);
                        }
                    }
            };

        }

    }

}
