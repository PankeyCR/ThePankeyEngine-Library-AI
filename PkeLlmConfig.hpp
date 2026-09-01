#pragma once

#include <cstdint>
#include <limits>

#include "PkeConfig.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            struct PkeLlmConfig : PkeConfig{
                uint32_t vocabularySize = 256;
                uint32_t contextLength = 128;
                uint32_t embeddingSize = 256;
                uint32_t layerCount = 4;
                uint32_t headCount = 4;
                uint32_t feedForwardSize = 1024;

                static uint64_t saturatingMultiply(uint64_t a_left, uint64_t a_right){
                    if(a_left == 0 || a_right == 0){
                        return 0;
                    }
                    const uint64_t i_maximum = std::numeric_limits<uint64_t>::max();
                    if(a_left > i_maximum / a_right){
                        return i_maximum;
                    }
                    return a_left * a_right;
                }

                static uint64_t saturatingAdd(uint64_t a_left, uint64_t a_right){
                    const uint64_t i_maximum = std::numeric_limits<uint64_t>::max();
                    if(a_left > i_maximum - a_right){
                        return i_maximum;
                    }
                    return a_left + a_right;
                }

                uint64_t estimatedParameterCount() const{
                    const uint64_t i_width = embeddingSize;
                    const uint64_t i_vocab = vocabularySize;
                    const uint64_t i_layers = layerCount;
                    const uint64_t i_ffn = feedForwardSize;
                    const uint64_t i_tokenEmbedding = saturatingMultiply(i_vocab, i_width);
                    const uint64_t i_positionEmbedding = saturatingMultiply(static_cast<uint64_t>(contextLength), i_width);
                    const uint64_t i_output = saturatingMultiply(i_width, i_vocab);
                    const uint64_t i_attention = saturatingMultiply(5ull, saturatingMultiply(i_width, i_width));
                    const uint64_t i_feedForward = saturatingMultiply(3ull, saturatingMultiply(i_width, i_ffn));
                    const uint64_t i_norm = saturatingMultiply(2ull, i_width);
                    const uint64_t i_perLayer = saturatingAdd(saturatingAdd(i_attention, i_feedForward), i_norm);
                    return saturatingAdd(
                        saturatingAdd(i_tokenEmbedding, i_positionEmbedding),
                        saturatingAdd(i_output, saturatingMultiply(i_layers, i_perLayer)));
                }

                uint64_t estimatedTrainingBytes() const{
                    const uint64_t i_parameters = estimatedParameterCount();
                    const uint64_t i_sequenceBytes = saturatingMultiply(
                        saturatingMultiply(
                            saturatingMultiply(static_cast<uint64_t>(microBatchSize), contextLength),
                            embeddingSize),
                        sizeof(float));
                    const uint64_t i_attentionBytes = saturatingMultiply(
                        saturatingMultiply(
                            saturatingMultiply(static_cast<uint64_t>(microBatchSize), headCount),
                            saturatingMultiply(contextLength, contextLength)),
                        sizeof(float));
                    const uint64_t i_parameterBytes = saturatingMultiply(i_parameters, 3ull * sizeof(float));
                    const uint64_t i_optimizerBytes = saturatingMultiply(i_parameters, 2ull);
                    return saturatingAdd(
                        saturatingAdd(i_parameterBytes, i_optimizerBytes),
                        saturatingAdd(i_sequenceBytes, i_attentionBytes));
                }
            };

        }

    }

}
