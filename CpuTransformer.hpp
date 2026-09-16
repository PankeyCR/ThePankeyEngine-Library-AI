#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "PkeLlmConfig.hpp"
#include "TokenSequence.hpp"
#include "Adam.hpp"
namespace pankey{

    namespace Library{

        namespace AI{

            class CpuTransformerThreadedTrainer;
            class CpuTransformerThreadedRuntime;
            class CudaTransformerRuntime;

            class CpuTransformer{
                public:
                    friend class CpuTransformerThreadedTrainer;
                    friend class CpuTransformerThreadedRuntime;
                    friend class CudaTransformerRuntime;
                    explicit CpuTransformer(const PkeLlmConfig& a_config)
                        : m_config(a_config){
                        validateConfig(m_config);
                        initialize();
                    }

                    const PkeLlmConfig& config() const{
                        return m_config;
                    }

                    std::vector<std::vector<float>> logitsBatch(const std::vector<TokenSequence>& a_batch) const{
                        std::vector<std::vector<float>> i_result;
                        i_result.reserve(a_batch.size());
                        for(const TokenSequence& i_tokens : a_batch){
                            i_result.push_back(logits(i_tokens));
                        }
                        return i_result;
                    }

                    bool save(const char* a_path) const{
                        std::ofstream i_file(a_path, std::ios::binary);
                        if(!i_file){
                            return false;
                        }
                        const uint32_t i_magic = 0x504B4541u;
                        const uint32_t i_version = 2;
                        i_file.write(reinterpret_cast<const char*>(&i_magic), sizeof(i_magic));
                        i_file.write(reinterpret_cast<const char*>(&i_version), sizeof(i_version));
                        i_file.write(reinterpret_cast<const char*>(&m_config), sizeof(m_config));
                        writeVector(i_file, m_embedding);
                        writeVector(i_file, m_position);
                        writeVector(i_file, m_output);
                        writeVector(i_file, m_outputBias);
                        writeVector(i_file, m_qkv);
                        writeVector(i_file, m_attentionOutput);
                        writeVector(i_file, m_attentionNorm);
                        writeVector(i_file, m_ffnGate);
                        writeVector(i_file, m_ffnUp);
                        writeVector(i_file, m_ffnDown);
                        writeVector(i_file, m_ffnNorm);
                        writeOptimizer(i_file, m_optEmbedding, m_embedding.size());
                        writeOptimizer(i_file, m_optPosition, m_position.size());
                        writeOptimizer(i_file, m_optOutput, m_output.size());
                        writeOptimizer(i_file, m_optOutputBias, m_outputBias.size());
                        writeOptimizer(i_file, m_optQkv, m_qkv.size());
                        writeOptimizer(i_file, m_optAttentionOutput, m_attentionOutput.size());
                        writeOptimizer(i_file, m_optAttentionNorm, m_attentionNorm.size());
                        writeOptimizer(i_file, m_optFfnGate, m_ffnGate.size());
                        writeOptimizer(i_file, m_optFfnUp, m_ffnUp.size());
                        writeOptimizer(i_file, m_optFfnDown, m_ffnDown.size());
                        writeOptimizer(i_file, m_optFfnNorm, m_ffnNorm.size());
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
                        i_file.read(reinterpret_cast<char*>(&i_magic), sizeof(i_magic));
                        i_file.read(reinterpret_cast<char*>(&i_version), sizeof(i_version));
                        i_file.read(reinterpret_cast<char*>(&i_config), sizeof(i_config));
                        if(!i_file || i_magic != 0x504B4541u || (i_version != 1 && i_version != 2)){
                            return false;
                        }
                        try{
                            validateConfig(i_config);
                        }catch(const std::invalid_argument&){
                            return false;
                        }
                        const ParameterSizes i_sizes = expectedParameterSizes(i_config);
                        std::vector<float> i_embedding;
                        std::vector<float> i_position;
                        std::vector<float> i_output;
                        std::vector<float> i_outputBias;
                        std::vector<float> i_qkv;
                        std::vector<float> i_attentionOutput;
                        std::vector<float> i_attentionNorm;
                        std::vector<float> i_ffnGate;
                        std::vector<float> i_ffnUp;
                        std::vector<float> i_ffnDown;
                        std::vector<float> i_ffnNorm;
                        if(!readVector(i_file, i_embedding, i_sizes.embedding)
                            || !readVector(i_file, i_position, i_sizes.position)
                            || !readVector(i_file, i_output, i_sizes.output)
                            || !readVector(i_file, i_outputBias, i_sizes.outputBias)
                            || !readVector(i_file, i_qkv, i_sizes.qkv)
                            || !readVector(i_file, i_attentionOutput, i_sizes.attentionOutput)
                            || !readVector(i_file, i_attentionNorm, i_sizes.attentionNorm)
                            || !readVector(i_file, i_ffnGate, i_sizes.ffnGate)
                            || !readVector(i_file, i_ffnUp, i_sizes.ffnUp)
                            || !readVector(i_file, i_ffnDown, i_sizes.ffnDown)
                            || !readVector(i_file, i_ffnNorm, i_sizes.ffnNorm)){
                            return false;
                        }
                        OptimizerState i_optEmbedding;
                        OptimizerState i_optPosition;
                        OptimizerState i_optOutput;
                        OptimizerState i_optOutputBias;
                        OptimizerState i_optQkv;
                        OptimizerState i_optAttentionOutput;
                        OptimizerState i_optAttentionNorm;
                        OptimizerState i_optFfnGate;
                        OptimizerState i_optFfnUp;
                        OptimizerState i_optFfnDown;
                        OptimizerState i_optFfnNorm;
                        if(i_version == 2){
                            if(!readOptimizer(i_file, i_sizes.embedding, i_optEmbedding)
                                || !readOptimizer(i_file, i_sizes.position, i_optPosition)
                                || !readOptimizer(i_file, i_sizes.output, i_optOutput)
                                || !readOptimizer(i_file, i_sizes.outputBias, i_optOutputBias)
                                || !readOptimizer(i_file, i_sizes.qkv, i_optQkv)
                                || !readOptimizer(i_file, i_sizes.attentionOutput, i_optAttentionOutput)
                                || !readOptimizer(i_file, i_sizes.attentionNorm, i_optAttentionNorm)
                                || !readOptimizer(i_file, i_sizes.ffnGate, i_optFfnGate)
                                || !readOptimizer(i_file, i_sizes.ffnUp, i_optFfnUp)
                                || !readOptimizer(i_file, i_sizes.ffnDown, i_optFfnDown)
                                || !readOptimizer(i_file, i_sizes.ffnNorm, i_optFfnNorm)){
                                return false;
                            }
                        }
                        m_embedding = i_embedding;
                        m_position = i_position;
                        m_output = i_output;
                        m_outputBias = i_outputBias;
                        m_qkv = i_qkv;
                        m_attentionOutput = i_attentionOutput;
                        m_attentionNorm = i_attentionNorm;
                        m_ffnGate = i_ffnGate;
                        m_ffnUp = i_ffnUp;
                        m_ffnDown = i_ffnDown;
                        m_ffnNorm = i_ffnNorm;
                        m_config = i_config;
                        resizeWorkspaces();
                        if(i_version == 2){
                            applyOptimizerState(m_optEmbedding, i_optEmbedding, i_sizes.embedding);
                            applyOptimizerState(m_optPosition, i_optPosition, i_sizes.position);
                            applyOptimizerState(m_optOutput, i_optOutput, i_sizes.output);
                            applyOptimizerState(m_optOutputBias, i_optOutputBias, i_sizes.outputBias);
                            applyOptimizerState(m_optQkv, i_optQkv, i_sizes.qkv);
                            applyOptimizerState(m_optAttentionOutput, i_optAttentionOutput, i_sizes.attentionOutput);
                            applyOptimizerState(m_optAttentionNorm, i_optAttentionNorm, i_sizes.attentionNorm);
                            applyOptimizerState(m_optFfnGate, i_optFfnGate, i_sizes.ffnGate);
                            applyOptimizerState(m_optFfnUp, i_optFfnUp, i_sizes.ffnUp);
                            applyOptimizerState(m_optFfnDown, i_optFfnDown, i_sizes.ffnDown);
                            applyOptimizerState(m_optFfnNorm, i_optFfnNorm, i_sizes.ffnNorm);
                        }else{
                            resetOptimizers();
                        }
                        if(!parametersFinite()){
                            resetOptimizers();
                            return false;
                        }
                        return true;
                    }

                    std::vector<float> logits(const TokenSequence& a_tokens) const{
                        return logits(sharedParameters(), m_inferenceWorkspace, a_tokens);
                    }

                    std::vector<float> probabilities(const TokenSequence& a_tokens) const{
                        return probabilities(sharedParameters(), m_inferenceWorkspace, a_tokens);
                    }

                    // Trains on a batch of next-token examples. Each TokenSequence of length L
                    // supplies L-1 supervised predictions (position p predicts token p+1) scored
                    // with cross-entropy. Performs one Adam step on the averaged batch gradient and
                    // returns the mean cross-entropy loss measured before the update.
                    float train(const std::vector<TokenSequence>& a_batch, float a_learningRate){
                        if(!std::isfinite(a_learningRate) || a_learningRate <= 0.0f){
                            throw std::invalid_argument("invalid transformer learning rate");
                        }
                        if(!parametersFinite()){
                            throw std::runtime_error("transformer parameters must stay finite");
                        }
                        TrainGradients& i_gradients = m_gradientWorkspace;
                        uint32_t i_count = 0;
                        const float i_loss = accumulateBatch(sharedParameters(),
                                                             m_trainingWorkspace,
                                                             a_batch,
                                                             i_gradients,
                                                             i_count);
                        if(i_count == 0){
                            return 0.0f;
                        }
                        const float i_scale = 1.0f / static_cast<float>(i_count);
                        scaleGradients(i_gradients, i_scale);
                        if(!clipGradients(i_gradients, 1.0f)){
                            throw std::runtime_error("non-finite transformer gradients");
                        }
                        applyOptimizer(i_gradients, a_learningRate);
                        if(!parametersFinite()){
                            throw std::runtime_error("non-finite transformer parameters after update");
                        }
                        const float i_meanLoss = i_loss * i_scale;
                        if(!std::isfinite(i_meanLoss)){
                            throw std::runtime_error("non-finite transformer mean loss");
                        }
                        return i_meanLoss;
                    }

                private:
                    PkeLlmConfig m_config;
                    std::vector<float> m_embedding;
                    std::vector<float> m_position;
                    std::vector<float> m_output;
                    std::vector<float> m_outputBias;
                    std::vector<float> m_qkv;
                    std::vector<float> m_attentionOutput;
                    std::vector<float> m_attentionNorm;
                    std::vector<float> m_ffnGate;
                    std::vector<float> m_ffnUp;
                    std::vector<float> m_ffnDown;
                    std::vector<float> m_ffnNorm;

                    Adam m_optEmbedding;
                    Adam m_optPosition;
                    Adam m_optOutput;
                    Adam m_optOutputBias;
                    Adam m_optQkv;
                    Adam m_optAttentionOutput;
                    Adam m_optAttentionNorm;
                    Adam m_optFfnGate;
                    Adam m_optFfnUp;
                    Adam m_optFfnDown;
                    Adam m_optFfnNorm;

                    struct TrainGradients{
                        std::vector<float> embedding;
                        std::vector<float> position;
                        std::vector<float> output;
                        std::vector<float> outputBias;
                        std::vector<float> qkv;
                        std::vector<float> attentionOutput;
                        std::vector<float> attentionNorm;
                        std::vector<float> ffnGate;
                        std::vector<float> ffnUp;
                        std::vector<float> ffnDown;
                        std::vector<float> ffnNorm;
                    };

                    struct SharedParameters{
                        const std::vector<float>* embedding = nullptr;
                        const std::vector<float>* position = nullptr;
                        const std::vector<float>* output = nullptr;
                        const std::vector<float>* outputBias = nullptr;
                        const std::vector<float>* qkv = nullptr;
                        const std::vector<float>* attentionOutput = nullptr;
                        const std::vector<float>* attentionNorm = nullptr;
                        const std::vector<float>* ffnGate = nullptr;
                        const std::vector<float>* ffnUp = nullptr;
                        const std::vector<float>* ffnDown = nullptr;
                        const std::vector<float>* ffnNorm = nullptr;
                    };

                    struct InferenceWorkspace{
                        std::vector<float> hidden;
                        std::vector<float> norm;
                        std::vector<float> qkv;
                        std::vector<float> attention;
                        std::vector<float> residual;
                        std::vector<float> scores;
                    };

                    struct ParameterSizes{
                        size_t embedding = 0;
                        size_t position = 0;
                        size_t output = 0;
                        size_t outputBias = 0;
                        size_t qkv = 0;
                        size_t attentionOutput = 0;
                        size_t attentionNorm = 0;
                        size_t ffnGate = 0;
                        size_t ffnUp = 0;
                        size_t ffnDown = 0;
                        size_t ffnNorm = 0;
                    };

                    struct OptimizerState{
                        uint64_t step = 0;
                        std::vector<float> first;
                        std::vector<float> second;
                    };

                    struct TrainingWorkspace{
                        std::vector<uint32_t> token;
                        std::vector<float> layerInput;
                        std::vector<float> rstd1;
                        std::vector<float> rstd2;
                        std::vector<float> xn1;
                        std::vector<float> xn2;
                        std::vector<float> qkvCache;
                        std::vector<float> attnWeights;
                        std::vector<float> context;
                        std::vector<float> hiddenMid;
                        std::vector<float> gate;
                        std::vector<float> up;
                        std::vector<float> logits;
                        std::vector<float> dFinal;
                        std::vector<float> dHidden;
                        std::vector<float> dNormB;
                        std::vector<float> dMid;
                        std::vector<float> dInput;
                        std::vector<float> dContext;
                        std::vector<float> dQkv;
                        std::vector<float> dNormA;
                        std::vector<float> dWeight;
                    };

                    mutable InferenceWorkspace m_inferenceWorkspace;
                    TrainGradients m_gradientWorkspace;
                    TrainingWorkspace m_trainingWorkspace;

                    static void validateConfig(const PkeLlmConfig& a_config){
                        if(a_config.vocabularySize == 0
                            || a_config.contextLength == 0
                            || a_config.microBatchSize == 0
                            || !std::isfinite(a_config.learningRate)
                            || a_config.learningRate <= 0.0f
                            || a_config.embeddingSize == 0
                            || a_config.layerCount == 0
                            || a_config.headCount == 0
                            || a_config.embeddingSize % a_config.headCount != 0
                            || a_config.feedForwardSize == 0
                            || a_config.memoryBudgetBytes == 0
                            || a_config.estimatedParameterCount() == std::numeric_limits<uint64_t>::max()
                            || a_config.estimatedTrainingBytes() == std::numeric_limits<uint64_t>::max()){
                            throw std::invalid_argument("invalid transformer configuration");
                        }
                    }

                    static bool isFiniteVector(const std::vector<float>& a_values){
                        for(float i_value : a_values){
                            if(!std::isfinite(i_value)){
                                return false;
                            }
                        }
                        return true;
                    }

                    static ParameterSizes expectedParameterSizes(const PkeLlmConfig& a_config){
                        ParameterSizes i_sizes;
                        const size_t i_width = static_cast<size_t>(a_config.embeddingSize);
                        const size_t i_vocab = static_cast<size_t>(a_config.vocabularySize);
                        const size_t i_context = static_cast<size_t>(a_config.contextLength);
                        const size_t i_layers = static_cast<size_t>(a_config.layerCount);
                        const size_t i_ffn = static_cast<size_t>(a_config.feedForwardSize);
                        i_sizes.embedding = i_vocab * i_width;
                        i_sizes.position = i_context * i_width;
                        i_sizes.output = i_width * i_vocab;
                        i_sizes.outputBias = i_vocab;
                        i_sizes.qkv = i_layers * 3u * i_width * i_width;
                        i_sizes.attentionOutput = i_layers * i_width * i_width;
                        i_sizes.attentionNorm = i_layers * i_width;
                        i_sizes.ffnGate = i_layers * i_width * i_ffn;
                        i_sizes.ffnUp = i_layers * i_width * i_ffn;
                        i_sizes.ffnDown = i_layers * i_ffn * i_width;
                        i_sizes.ffnNorm = i_layers * i_width;
                        return i_sizes;
                    }

                    static void writeVector(std::ofstream& a_file, const std::vector<float>& a_values){
                        const uint64_t i_size = static_cast<uint64_t>(a_values.size());
                        a_file.write(reinterpret_cast<const char*>(&i_size), sizeof(i_size));
                        if(i_size != 0){
                            a_file.write(reinterpret_cast<const char*>(a_values.data()),
                                         static_cast<std::streamsize>(i_size * sizeof(float)));
                        }
                    }

                    static bool readVector(std::ifstream& a_file,
                                           std::vector<float>& a_values,
                                           size_t a_expectedSize){
                        uint64_t i_size = 0;
                        a_file.read(reinterpret_cast<char*>(&i_size), sizeof(i_size));
                        if(!a_file || i_size != static_cast<uint64_t>(a_expectedSize)){
                            return false;
                        }
                        a_values.resize(static_cast<size_t>(i_size));
                        if(i_size != 0){
                            a_file.read(reinterpret_cast<char*>(a_values.data()),
                                        static_cast<std::streamsize>(i_size * sizeof(float)));
                        }
                        return static_cast<bool>(a_file) && isFiniteVector(a_values);
                    }

                    static void writeOptimizer(std::ofstream& a_file,
                                               const Adam& a_adam,
                                               size_t a_expectedSize){
                        const uint64_t i_step = a_adam.step();
                        a_file.write(reinterpret_cast<const char*>(&i_step), sizeof(i_step));
                        const std::vector<float>& i_first = a_adam.firstMoment();
                        const std::vector<float>& i_second = a_adam.secondMoment();
                        if(i_step == 0 && i_first.empty() && i_second.empty()){
                            const std::vector<float> i_zeroMoments(a_expectedSize, 0.0f);
                            writeVector(a_file, i_zeroMoments);
                            writeVector(a_file, i_zeroMoments);
                            return;
                        }
                        writeVector(a_file, i_first);
                        writeVector(a_file, i_second);
                    }

                    static bool readOptimizer(std::ifstream& a_file,
                                              size_t a_expectedSize,
                                              OptimizerState& a_state){
                        a_file.read(reinterpret_cast<char*>(&a_state.step), sizeof(a_state.step));
                        if(!a_file){
                            return false;
                        }
                        if(!readVector(a_file, a_state.first, a_expectedSize)
                            || !readVector(a_file, a_state.second, a_expectedSize)){
                            return false;
                        }
                        for(float i_value : a_state.second){
                            if(i_value < 0.0f){
                                return false;
                            }
                        }
                        return true;
                    }

                    static void applyOptimizerState(Adam& a_adam,
                                                    const OptimizerState& a_state,
                                                    size_t a_expectedSize){
                        if(a_state.first.size() != a_expectedSize || a_state.second.size() != a_expectedSize){
                            throw std::runtime_error("invalid transformer optimizer state");
                        }
                        a_adam.loadState(a_state.step, a_state.first, a_state.second);
                    }

                    void resetOptimizers(){
                        m_optEmbedding.reset(0);
                        m_optPosition.reset(0);
                        m_optOutput.reset(0);
                        m_optOutputBias.reset(0);
                        m_optQkv.reset(0);
                        m_optAttentionOutput.reset(0);
                        m_optAttentionNorm.reset(0);
                        m_optFfnGate.reset(0);
                        m_optFfnUp.reset(0);
                        m_optFfnDown.reset(0);
                        m_optFfnNorm.reset(0);
                    }

                    void initialize(){
                        const uint32_t i_width = m_config.embeddingSize;
                        const uint32_t i_headWidth = i_width / m_config.headCount;
                        const size_t i_layerSize = static_cast<size_t>(m_config.layerCount);
                        m_embedding.resize(static_cast<size_t>(m_config.vocabularySize) * i_width);
                        m_position.resize(static_cast<size_t>(m_config.contextLength) * i_width);
                        m_output.resize(static_cast<size_t>(i_width) * m_config.vocabularySize);
                        m_outputBias.resize(m_config.vocabularySize, 0.0f);
                        m_qkv.resize(i_layerSize * 3 * i_width * i_width);
                        m_attentionOutput.resize(i_layerSize * i_width * i_width);
                        m_attentionNorm.resize(i_layerSize * i_width, 1.0f);
                        m_ffnGate.resize(i_layerSize * i_width * m_config.feedForwardSize);
                        m_ffnUp.resize(i_layerSize * i_width * m_config.feedForwardSize);
                        m_ffnDown.resize(i_layerSize * m_config.feedForwardSize * i_width);
                        m_ffnNorm.resize(i_layerSize * i_width, 1.0f);

                        std::mt19937 i_generator(m_config.seed);
                        std::normal_distribution<float> i_distribution(0.0f, 0.02f);
                        for(float& i_value : m_embedding){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_position){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_output){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_qkv){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_attentionOutput){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_ffnGate){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_ffnUp){ i_value = i_distribution(i_generator); }
                        for(float& i_value : m_ffnDown){ i_value = i_distribution(i_generator); }
                        resizeWorkspaces();
                        resetOptimizers();
                        (void)i_headWidth;
                    }

                    static void zeroVector(std::vector<float>& a_values){
                        std::fill(a_values.begin(), a_values.end(), 0.0f);
                    }

                    static void zeroVectorPrefix(std::vector<float>& a_values, size_t a_size){
                        std::fill(a_values.begin(), a_values.begin() + static_cast<std::ptrdiff_t>(a_size), 0.0f);
                    }

                    static float* workspaceLayer(std::vector<float>& a_storage,
                                                 size_t a_stride,
                                                 uint32_t a_layer){
                        return a_storage.data() + static_cast<size_t>(a_layer) * a_stride;
                    }

                    static const float* workspaceLayer(const std::vector<float>& a_storage,
                                                       size_t a_stride,
                                                       uint32_t a_layer){
                        return a_storage.data() + static_cast<size_t>(a_layer) * a_stride;
                    }

                    SharedParameters sharedParameters() const{
                        SharedParameters i_parameters;
                        i_parameters.embedding = &m_embedding;
                        i_parameters.position = &m_position;
                        i_parameters.output = &m_output;
                        i_parameters.outputBias = &m_outputBias;
                        i_parameters.qkv = &m_qkv;
                        i_parameters.attentionOutput = &m_attentionOutput;
                        i_parameters.attentionNorm = &m_attentionNorm;
                        i_parameters.ffnGate = &m_ffnGate;
                        i_parameters.ffnUp = &m_ffnUp;
                        i_parameters.ffnDown = &m_ffnDown;
                        i_parameters.ffnNorm = &m_ffnNorm;
                        return i_parameters;
                    }

                    static void resizeGradientWorkspace(TrainGradients& a_gradients,
                                                        const ParameterSizes& a_sizes){
                        a_gradients.embedding.resize(a_sizes.embedding);
                        a_gradients.position.resize(a_sizes.position);
                        a_gradients.output.resize(a_sizes.output);
                        a_gradients.outputBias.resize(a_sizes.outputBias);
                        a_gradients.qkv.resize(a_sizes.qkv);
                        a_gradients.attentionOutput.resize(a_sizes.attentionOutput);
                        a_gradients.attentionNorm.resize(a_sizes.attentionNorm);
                        a_gradients.ffnGate.resize(a_sizes.ffnGate);
                        a_gradients.ffnUp.resize(a_sizes.ffnUp);
                        a_gradients.ffnDown.resize(a_sizes.ffnDown);
                        a_gradients.ffnNorm.resize(a_sizes.ffnNorm);
                    }

                    static void zeroGradients(TrainGradients& a_gradients){
                        zeroVector(a_gradients.embedding);
                        zeroVector(a_gradients.position);
                        zeroVector(a_gradients.output);
                        zeroVector(a_gradients.outputBias);
                        zeroVector(a_gradients.qkv);
                        zeroVector(a_gradients.attentionOutput);
                        zeroVector(a_gradients.attentionNorm);
                        zeroVector(a_gradients.ffnGate);
                        zeroVector(a_gradients.ffnUp);
                        zeroVector(a_gradients.ffnDown);
                        zeroVector(a_gradients.ffnNorm);
                    }

                    static void addGradients(TrainGradients& a_target,
                                             const TrainGradients& a_source){
                        const auto i_add = [](std::vector<float>& a_destination,
                                              const std::vector<float>& a_values){
                            for(size_t i = 0; i < a_destination.size(); ++i){
                                a_destination[i] += a_values[i];
                            }
                        };
                        i_add(a_target.embedding, a_source.embedding);
                        i_add(a_target.position, a_source.position);
                        i_add(a_target.output, a_source.output);
                        i_add(a_target.outputBias, a_source.outputBias);
                        i_add(a_target.qkv, a_source.qkv);
                        i_add(a_target.attentionOutput, a_source.attentionOutput);
                        i_add(a_target.attentionNorm, a_source.attentionNorm);
                        i_add(a_target.ffnGate, a_source.ffnGate);
                        i_add(a_target.ffnUp, a_source.ffnUp);
                        i_add(a_target.ffnDown, a_source.ffnDown);
                        i_add(a_target.ffnNorm, a_source.ffnNorm);
                    }

                    static void resizeTrainingWorkspace(TrainingWorkspace& a_workspace,
                                                        const PkeLlmConfig& a_config){
                        const size_t i_layers = static_cast<size_t>(a_config.layerCount);
                        const size_t i_width = static_cast<size_t>(a_config.embeddingSize);
                        const size_t i_length = static_cast<size_t>(a_config.contextLength);
                        const size_t i_heads = static_cast<size_t>(a_config.headCount);
                        const size_t i_ffn = static_cast<size_t>(a_config.feedForwardSize);
                        const size_t i_hiddenSize = i_length * i_width;
                        const size_t i_qkvSize = i_hiddenSize * 3u;
                        const size_t i_attnWeightSize = i_heads * i_length * i_length;
                        const size_t i_ffnSize = i_length * i_ffn;

                        a_workspace.token.resize(i_length);
                        a_workspace.layerInput.resize((i_layers + 1u) * i_hiddenSize);
                        a_workspace.rstd1.resize(i_layers * i_length);
                        a_workspace.rstd2.resize(i_layers * i_length);
                        a_workspace.xn1.resize(i_layers * i_hiddenSize);
                        a_workspace.xn2.resize(i_layers * i_hiddenSize);
                        a_workspace.qkvCache.resize(i_layers * i_qkvSize);
                        a_workspace.attnWeights.resize(i_layers * i_attnWeightSize);
                        a_workspace.context.resize(i_layers * i_hiddenSize);
                        a_workspace.hiddenMid.resize(i_layers * i_hiddenSize);
                        a_workspace.gate.resize(i_layers * i_ffnSize);
                        a_workspace.up.resize(i_layers * i_ffnSize);
                        a_workspace.logits.resize(static_cast<size_t>(a_config.vocabularySize));
                        a_workspace.dFinal.resize(i_hiddenSize);
                        a_workspace.dHidden.resize(i_hiddenSize);
                        a_workspace.dNormB.resize(i_hiddenSize);
                        a_workspace.dMid.resize(i_hiddenSize);
                        a_workspace.dInput.resize(i_hiddenSize);
                        a_workspace.dContext.resize(i_hiddenSize);
                        a_workspace.dQkv.resize(i_qkvSize);
                        a_workspace.dNormA.resize(i_hiddenSize);
                        a_workspace.dWeight.resize(i_length);
                    }

                    static void resizeInferenceWorkspace(InferenceWorkspace& a_workspace,
                                                         const PkeLlmConfig& a_config){
                        const size_t i_width = static_cast<size_t>(a_config.embeddingSize);
                        const size_t i_length = static_cast<size_t>(a_config.contextLength);
                        const size_t i_hiddenSize = i_length * i_width;
                        a_workspace.hidden.resize(i_hiddenSize);
                        a_workspace.norm.resize(i_hiddenSize);
                        a_workspace.qkv.resize(i_hiddenSize * 3u);
                        a_workspace.attention.resize(i_hiddenSize);
                        a_workspace.residual.resize(i_hiddenSize);
                        a_workspace.scores.resize(i_length);
                    }

                    void resizeWorkspaces(){
                        const ParameterSizes i_sizes = expectedParameterSizes(m_config);
                        resizeInferenceWorkspace(m_inferenceWorkspace, m_config);
                        resizeGradientWorkspace(m_gradientWorkspace, i_sizes);
                        resizeTrainingWorkspace(m_trainingWorkspace, m_config);
                    }

                    bool parametersFinite() const{
                        return isFiniteVector(m_embedding)
                            && isFiniteVector(m_position)
                            && isFiniteVector(m_output)
                            && isFiniteVector(m_outputBias)
                            && isFiniteVector(m_qkv)
                            && isFiniteVector(m_attentionOutput)
                            && isFiniteVector(m_attentionNorm)
                            && isFiniteVector(m_ffnGate)
                            && isFiniteVector(m_ffnUp)
                            && isFiniteVector(m_ffnDown)
                            && isFiniteVector(m_ffnNorm);
                    }

                    static float silu(float a_value){
                        return a_value / (1.0f + std::exp(-a_value));
                    }

                    void normalize(std::vector<float>& a_values,
                                   uint32_t a_offset,
                                   uint32_t a_width,
                                   const std::vector<float>& a_scale,
                                   uint32_t a_scaleOffset) const{
                        float i_mean = 0.0f;
                        for(uint32_t i = 0; i < a_width; ++i){
                            i_mean += a_values[a_offset + i] * a_values[a_offset + i];
                        }
                        const float i_factor = 1.0f / std::sqrt(i_mean / static_cast<float>(a_width) + 1.0e-5f);
                        for(uint32_t i = 0; i < a_width; ++i){
                            a_values[a_offset + i] *= i_factor * a_scale[a_scaleOffset + i];
                        }
                    }

                    void runLayer(const SharedParameters& a_parameters,
                                  InferenceWorkspace& a_workspace,
                                  uint32_t a_layer,
                                  std::vector<float>& a_hidden,
                                  uint32_t a_length) const{
                        const uint32_t i_width = m_config.embeddingSize;
                        const uint32_t i_headWidth = i_width / m_config.headCount;
                        const size_t i_hiddenSize = static_cast<size_t>(a_length) * i_width;
                        const size_t i_qkvBase = static_cast<size_t>(a_layer) * 3 * i_width * i_width;
                        const size_t i_attentionBase = static_cast<size_t>(a_layer) * i_width * i_width;
                        const size_t i_gateBase = static_cast<size_t>(a_layer) * i_width * m_config.feedForwardSize;
                        const size_t i_downBase = static_cast<size_t>(a_layer) * m_config.feedForwardSize * i_width;
                        const std::vector<float>& i_qkvWeights = *a_parameters.qkv;
                        const std::vector<float>& i_attentionOutputWeights = *a_parameters.attentionOutput;
                        const std::vector<float>& i_attentionNormWeights = *a_parameters.attentionNorm;
                        const std::vector<float>& i_ffnGateWeights = *a_parameters.ffnGate;
                        const std::vector<float>& i_ffnUpWeights = *a_parameters.ffnUp;
                        const std::vector<float>& i_ffnDownWeights = *a_parameters.ffnDown;
                        const std::vector<float>& i_ffnNormWeights = *a_parameters.ffnNorm;
                        std::vector<float>& i_norm = a_workspace.norm;
                        std::copy_n(a_hidden.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_norm.begin());
                        for(uint32_t i_position = 0; i_position < a_length; ++i_position){
                            normalize(i_norm, i_position * i_width, i_width, i_attentionNormWeights, a_layer * i_width);
                        }

                        std::vector<float>& i_qkv = a_workspace.qkv;
                        for(uint32_t i_position = 0; i_position < a_length; ++i_position){
                            for(uint32_t i_output = 0; i_output < 3 * i_width; ++i_output){
                                float i_value = 0.0f;
                                for(uint32_t i_input = 0; i_input < i_width; ++i_input){
                                    i_value += i_norm[i_position * i_width + i_input]
                                        * i_qkvWeights[i_qkvBase + i_output * i_width + i_input];
                                }
                                i_qkv[i_position * 3 * i_width + i_output] = i_value;
                            }
                        }

                        std::vector<float>& i_attention = a_workspace.attention;
                        zeroVectorPrefix(i_attention, i_hiddenSize);
                        for(uint32_t i_head = 0; i_head < m_config.headCount; ++i_head){
                            for(uint32_t i_target = 0; i_target < a_length; ++i_target){
                                std::vector<float>& i_scores = a_workspace.scores;
                                float i_sum = 0.0f;
                                for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                    i_scores[i_source] = 0.0f;
                                    for(uint32_t i = 0; i < i_headWidth; ++i){
                                        i_scores[i_source] += i_qkv[i_target * 3 * i_width + i_head * i_headWidth + i]
                                            * i_qkv[i_source * 3 * i_width + i_width + i_head * i_headWidth + i];
                                    }
                                    i_scores[i_source] = std::exp(i_scores[i_source] / std::sqrt(static_cast<float>(i_headWidth)));
                                    i_sum += i_scores[i_source];
                                }
                                if(!std::isfinite(i_sum) || i_sum <= 0.0f){
                                    throw std::runtime_error("non-finite attention weights");
                                }
                                for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                    const float i_weight = i_scores[i_source] / i_sum;
                                    for(uint32_t i = 0; i < i_headWidth; ++i){
                                        i_attention[i_target * i_width + i_head * i_headWidth + i]
                                            += i_weight * i_qkv[i_source * 3 * i_width + 2 * i_width + i_head * i_headWidth + i];
                                    }
                                }
                            }
                        }

                        std::vector<float>& i_residual = a_workspace.residual;
                        std::copy_n(a_hidden.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_residual.begin());
                        for(uint32_t i_position = 0; i_position < a_length; ++i_position){
                            for(uint32_t i_output = 0; i_output < i_width; ++i_output){
                                float i_value = 0.0f;
                                for(uint32_t i_input = 0; i_input < i_width; ++i_input){
                                    i_value += i_attention[i_position * i_width + i_input]
                                        * i_attentionOutputWeights[i_attentionBase + i_output * i_width + i_input];
                                }
                                a_hidden[i_position * i_width + i_output] = i_residual[i_position * i_width + i_output] + i_value;
                            }
                        }

                        std::copy_n(a_hidden.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_norm.begin());
                        for(uint32_t i_position = 0; i_position < a_length; ++i_position){
                            normalize(i_norm, i_position * i_width, i_width, i_ffnNormWeights, a_layer * i_width);
                            for(uint32_t i_output = 0; i_output < m_config.feedForwardSize; ++i_output){
                                float i_gate = 0.0f;
                                float i_up = 0.0f;
                                for(uint32_t i_input = 0; i_input < i_width; ++i_input){
                                    i_gate += i_norm[i_position * i_width + i_input]
                                        * i_ffnGateWeights[i_gateBase + i_output * i_width + i_input];
                                    i_up += i_norm[i_position * i_width + i_input]
                                        * i_ffnUpWeights[i_gateBase + i_output * i_width + i_input];
                                }
                                const float i_value = silu(i_gate) * i_up;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    a_hidden[i_position * i_width + i] += i_value
                                        * i_ffnDownWeights[i_downBase + i * m_config.feedForwardSize + i_output];
                                }
                            }
                        }
                    }

                    static float sigmoid(float a_value){
                        return 1.0f / (1.0f + std::exp(-a_value));
                    }

                    static void scaleGradients(TrainGradients& a_gradients, float a_scale){
                        const auto i_scale = [a_scale](std::vector<float>& a_values){
                            for(float& i_value : a_values){
                                i_value *= a_scale;
                            }
                        };
                        i_scale(a_gradients.embedding);
                        i_scale(a_gradients.position);
                        i_scale(a_gradients.output);
                        i_scale(a_gradients.outputBias);
                        i_scale(a_gradients.qkv);
                        i_scale(a_gradients.attentionOutput);
                        i_scale(a_gradients.attentionNorm);
                        i_scale(a_gradients.ffnGate);
                        i_scale(a_gradients.ffnUp);
                        i_scale(a_gradients.ffnDown);
                        i_scale(a_gradients.ffnNorm);
                    }

                    static void scaleVector(std::vector<float>& a_values, float a_scale){
                        for(float& i_value : a_values){
                            i_value *= a_scale;
                        }
                    }

                    static bool clipGradients(TrainGradients& a_gradients, float a_maxNorm){
                        if(!std::isfinite(a_maxNorm) || a_maxNorm <= 0.0f){
                            return false;
                        }
                        double i_sumSquares = 0.0;
                        const auto i_accumulate = [&i_sumSquares](const std::vector<float>& a_values){
                            for(float i_value : a_values){
                                if(!std::isfinite(i_value)){
                                    return false;
                                }
                                i_sumSquares += static_cast<double>(i_value) * static_cast<double>(i_value);
                            }
                            return true;
                        };
                        if(!i_accumulate(a_gradients.embedding)
                            || !i_accumulate(a_gradients.position)
                            || !i_accumulate(a_gradients.output)
                            || !i_accumulate(a_gradients.outputBias)
                            || !i_accumulate(a_gradients.qkv)
                            || !i_accumulate(a_gradients.attentionOutput)
                            || !i_accumulate(a_gradients.attentionNorm)
                            || !i_accumulate(a_gradients.ffnGate)
                            || !i_accumulate(a_gradients.ffnUp)
                            || !i_accumulate(a_gradients.ffnDown)
                            || !i_accumulate(a_gradients.ffnNorm)){
                            return false;
                        }
                        const double i_norm = std::sqrt(i_sumSquares);
                        if(!std::isfinite(i_norm)){
                            return false;
                        }
                        if(i_norm <= static_cast<double>(a_maxNorm) || i_norm == 0.0){
                            return true;
                        }
                        const float i_scale = a_maxNorm / static_cast<float>(i_norm);
                        scaleVector(a_gradients.embedding, i_scale);
                        scaleVector(a_gradients.position, i_scale);
                        scaleVector(a_gradients.output, i_scale);
                        scaleVector(a_gradients.outputBias, i_scale);
                        scaleVector(a_gradients.qkv, i_scale);
                        scaleVector(a_gradients.attentionOutput, i_scale);
                        scaleVector(a_gradients.attentionNorm, i_scale);
                        scaleVector(a_gradients.ffnGate, i_scale);
                        scaleVector(a_gradients.ffnUp, i_scale);
                        scaleVector(a_gradients.ffnDown, i_scale);
                        scaleVector(a_gradients.ffnNorm, i_scale);
                        return true;
                    }

                    void applyOptimizer(const TrainGradients& a_gradients, float a_learningRate){
                        m_optEmbedding.setLearningRate(a_learningRate);
                        m_optPosition.setLearningRate(a_learningRate);
                        m_optOutput.setLearningRate(a_learningRate);
                        m_optOutputBias.setLearningRate(a_learningRate);
                        m_optQkv.setLearningRate(a_learningRate);
                        m_optAttentionOutput.setLearningRate(a_learningRate);
                        m_optAttentionNorm.setLearningRate(a_learningRate);
                        m_optFfnGate.setLearningRate(a_learningRate);
                        m_optFfnUp.setLearningRate(a_learningRate);
                        m_optFfnDown.setLearningRate(a_learningRate);
                        m_optFfnNorm.setLearningRate(a_learningRate);

                        if(!m_optEmbedding.update(m_embedding, a_gradients.embedding)
                            || !m_optPosition.update(m_position, a_gradients.position)
                            || !m_optOutput.update(m_output, a_gradients.output)
                            || !m_optOutputBias.update(m_outputBias, a_gradients.outputBias)
                            || !m_optQkv.update(m_qkv, a_gradients.qkv)
                            || !m_optAttentionOutput.update(m_attentionOutput, a_gradients.attentionOutput)
                            || !m_optAttentionNorm.update(m_attentionNorm, a_gradients.attentionNorm)
                            || !m_optFfnGate.update(m_ffnGate, a_gradients.ffnGate)
                            || !m_optFfnUp.update(m_ffnUp, a_gradients.ffnUp)
                            || !m_optFfnDown.update(m_ffnDown, a_gradients.ffnDown)
                            || !m_optFfnNorm.update(m_ffnNorm, a_gradients.ffnNorm)){
                            throw std::runtime_error("transformer optimizer update failed");
                        }
                    }

                    std::vector<float> logits(const SharedParameters& a_parameters,
                                              InferenceWorkspace& a_workspace,
                                              const TokenSequence& a_tokens) const{
                        const std::vector<float>& i_embedding = *a_parameters.embedding;
                        const std::vector<float>& i_positionWeights = *a_parameters.position;
                        const std::vector<float>& i_outputWeights = *a_parameters.output;
                        const std::vector<float>& i_outputBiasWeights = *a_parameters.outputBias;
                        const uint32_t i_length = std::min(a_tokens.size(), m_config.contextLength);
                        const uint32_t i_width = m_config.embeddingSize;
                        std::vector<float>& i_hidden = a_workspace.hidden;
                        for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                            const uint32_t i_token = a_tokens.get(i_position) % m_config.vocabularySize;
                            for(uint32_t i = 0; i < i_width; ++i){
                                i_hidden[i_position * i_width + i] = i_embedding[i_token * i_width + i]
                                    + i_positionWeights[i_position * i_width + i];
                            }
                        }

                        for(uint32_t i_layer = 0; i_layer < m_config.layerCount; ++i_layer){
                            runLayer(a_parameters, a_workspace, i_layer, i_hidden, i_length);
                        }

                        std::vector<float> i_result(m_config.vocabularySize, 0.0f);
                        if(i_length == 0){
                            return i_result;
                        }
                        const uint32_t i_last = (i_length - 1) * i_width;
                        for(uint32_t i_token = 0; i_token < m_config.vocabularySize; ++i_token){
                            float i_value = i_outputBiasWeights[i_token];
                            for(uint32_t i = 0; i < i_width; ++i){
                                i_value += i_hidden[i_last + i] * i_outputWeights[i * m_config.vocabularySize + i_token];
                            }
                            i_result[i_token] = i_value;
                        }
                        if(!isFiniteVector(i_result)){
                            throw std::runtime_error("non-finite transformer logits");
                        }
                        return i_result;
                    }

                    std::vector<float> probabilities(const SharedParameters& a_parameters,
                                                     InferenceWorkspace& a_workspace,
                                                     const TokenSequence& a_tokens) const{
                        std::vector<float> i_values = logits(a_parameters, a_workspace, a_tokens);
                        if(i_values.empty()){
                            return i_values;
                        }
                        const float i_maximum = *std::max_element(i_values.begin(), i_values.end());
                        float i_sum = 0.0f;
                        for(float& i_value : i_values){
                            i_value = std::exp(i_value - i_maximum);
                            i_sum += i_value;
                        }
                        if(!std::isfinite(i_sum) || i_sum <= 0.0f){
                            return std::vector<float>(i_values.size(), 0.0f);
                        }
                        for(float& i_value : i_values){
                            i_value /= i_sum;
                        }
                        if(!isFiniteVector(i_values)){
                            throw std::runtime_error("non-finite transformer probabilities");
                        }
                        return i_values;
                    }

                    float accumulateBatch(const SharedParameters& a_parameters,
                                          TrainingWorkspace& a_workspace,
                                          const std::vector<TokenSequence>& a_batch,
                                          TrainGradients& a_gradients,
                                          uint32_t& a_count) const{
                        zeroGradients(a_gradients);
                        float i_loss = 0.0f;
                        const size_t i_microBatchSize = std::max<size_t>(1, static_cast<size_t>(m_config.microBatchSize));
                        for(size_t i_begin = 0; i_begin < a_batch.size(); i_begin += i_microBatchSize){
                            const size_t i_end = std::min(a_batch.size(), i_begin + i_microBatchSize);
                            for(size_t i = i_begin; i < i_end; ++i){
                                const float i_sequenceLoss = accumulateSequence(a_parameters, a_workspace, a_batch[i], a_gradients, a_count);
                                if(!std::isfinite(i_sequenceLoss)){
                                    throw std::runtime_error("non-finite transformer loss");
                                }
                                i_loss += i_sequenceLoss;
                            }
                        }
                        return i_loss;
                    }

                    // Accumulates the cross-entropy gradient of a single sequence into a_gradients and
                    // returns the summed loss over the L-1 predicted positions. a_count is advanced by
                    // the number of predicted positions so the caller can average.
                    float accumulateSequence(const SharedParameters& a_parameters,
                                             TrainingWorkspace& a_workspace,
                                             const TokenSequence& a_tokens,
                                             TrainGradients& a_gradients,
                                             uint32_t& a_count) const{
                        const std::vector<float>& i_embedding = *a_parameters.embedding;
                        const std::vector<float>& i_positionWeights = *a_parameters.position;
                        const std::vector<float>& i_outputWeights = *a_parameters.output;
                        const std::vector<float>& i_outputBiasWeights = *a_parameters.outputBias;
                        const std::vector<float>& i_qkvWeights = *a_parameters.qkv;
                        const std::vector<float>& i_attentionOutputWeights = *a_parameters.attentionOutput;
                        const std::vector<float>& i_attentionNormWeights = *a_parameters.attentionNorm;
                        const std::vector<float>& i_ffnGateWeights = *a_parameters.ffnGate;
                        const std::vector<float>& i_ffnUpWeights = *a_parameters.ffnUp;
                        const std::vector<float>& i_ffnDownWeights = *a_parameters.ffnDown;
                        const std::vector<float>& i_ffnNormWeights = *a_parameters.ffnNorm;
                        const uint32_t i_width = m_config.embeddingSize;
                        const uint32_t i_headWidth = i_width / m_config.headCount;
                        const uint32_t i_heads = m_config.headCount;
                        const uint32_t i_ffn = m_config.feedForwardSize;
                        const uint32_t i_vocab = m_config.vocabularySize;
                        const uint32_t i_layers = m_config.layerCount;
                        const uint32_t i_length = std::min(a_tokens.size(), m_config.contextLength);
                        if(i_length < 2){
                            return 0.0f;
                        }

                        std::vector<uint32_t>& i_token = a_workspace.token;
                        for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                            i_token[i_position] = a_tokens.get(i_position) % i_vocab;
                        }

                        const size_t i_hiddenSize = static_cast<size_t>(i_length) * i_width;
                        const size_t i_qkvSize = i_hiddenSize * 3u;
                        const size_t i_hiddenStride = i_hiddenSize;
                        const size_t i_rstdStride = static_cast<size_t>(i_length);
                        const size_t i_attnWeightStride = static_cast<size_t>(i_heads) * i_length * i_length;
                        const size_t i_ffnStride = static_cast<size_t>(i_length) * i_ffn;

                        std::vector<float>& i_layerInputStorage = a_workspace.layerInput;
                        std::vector<float>& i_rstd1Storage = a_workspace.rstd1;
                        std::vector<float>& i_rstd2Storage = a_workspace.rstd2;
                        std::vector<float>& i_xn1Storage = a_workspace.xn1;
                        std::vector<float>& i_xn2Storage = a_workspace.xn2;
                        std::vector<float>& i_qkvStorage = a_workspace.qkvCache;
                        std::vector<float>& i_attnWeightStorage = a_workspace.attnWeights;
                        std::vector<float>& i_contextStorage = a_workspace.context;
                        std::vector<float>& i_hiddenMidStorage = a_workspace.hiddenMid;
                        std::vector<float>& i_gateStorage = a_workspace.gate;
                        std::vector<float>& i_upStorage = a_workspace.up;
                        float* const i_firstLayerInput = workspaceLayer(i_layerInputStorage, i_hiddenStride, 0);

                        for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                            for(uint32_t i = 0; i < i_width; ++i){
                                i_firstLayerInput[i_position * i_width + i] =
                                    i_embedding[i_token[i_position] * i_width + i]
                                    + i_positionWeights[i_position * i_width + i];
                            }
                        }

                        const float i_attnScale = 1.0f / std::sqrt(static_cast<float>(i_headWidth));

                        for(uint32_t i_layer = 0; i_layer < i_layers; ++i_layer){
                            const float* const i_input = workspaceLayer(i_layerInputStorage, i_hiddenStride, i_layer);
                            float* const i_rstd1Layer = workspaceLayer(i_rstd1Storage, i_rstdStride, i_layer);
                            float* const i_rstd2Layer = workspaceLayer(i_rstd2Storage, i_rstdStride, i_layer);
                            float* const i_normA = workspaceLayer(i_xn1Storage, i_hiddenStride, i_layer);
                            float* const i_normB = workspaceLayer(i_xn2Storage, i_hiddenStride, i_layer);
                            float* const i_qkv = workspaceLayer(i_qkvStorage, i_qkvSize, i_layer);
                            float* const i_attnWeightLayer = workspaceLayer(i_attnWeightStorage, i_attnWeightStride, i_layer);
                            float* const i_attention = workspaceLayer(i_contextStorage, i_hiddenStride, i_layer);
                            float* const i_mid = workspaceLayer(i_hiddenMidStorage, i_hiddenStride, i_layer);
                            float* const i_gateLayer = workspaceLayer(i_gateStorage, i_ffnStride, i_layer);
                            float* const i_upLayer = workspaceLayer(i_upStorage, i_ffnStride, i_layer);
                            float* const i_output = workspaceLayer(i_layerInputStorage, i_hiddenStride, i_layer + 1u);
                            const size_t i_qkvBase = static_cast<size_t>(i_layer) * 3 * i_width * i_width;
                            const size_t i_attentionBase = static_cast<size_t>(i_layer) * i_width * i_width;
                            const size_t i_gateBase = static_cast<size_t>(i_layer) * i_width * i_ffn;
                            const size_t i_downBase = static_cast<size_t>(i_layer) * i_ffn * i_width;

                            // RMS normalization before attention.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                float i_mean = 0.0f;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    const float i_value = i_input[i_position * i_width + i];
                                    i_mean += i_value * i_value;
                                }
                                const float i_rstd = 1.0f / std::sqrt(i_mean / static_cast<float>(i_width) + 1.0e-5f);
                                i_rstd1Layer[i_position] = i_rstd;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    i_normA[i_position * i_width + i] = i_input[i_position * i_width + i]
                                        * i_rstd * i_attentionNormWeights[i_layer * i_width + i];
                                }
                            }

                            // QKV projection.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                for(uint32_t i_out = 0; i_out < 3 * i_width; ++i_out){
                                    float i_value = 0.0f;
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        i_value += i_normA[i_position * i_width + i_in]
                                            * i_qkvWeights[i_qkvBase + i_out * i_width + i_in];
                                    }
                                    i_qkv[i_position * 3 * i_width + i_out] = i_value;
                                }
                            }

                            // Causal attention with cached softmax weights.
                            zeroVectorPrefix(i_contextStorage, static_cast<size_t>(i_layer) * i_hiddenStride + i_hiddenSize);
                            std::fill(i_attention, i_attention + static_cast<std::ptrdiff_t>(i_hiddenSize), 0.0f);
                            for(uint32_t i_head = 0; i_head < i_heads; ++i_head){
                                for(uint32_t i_target = 0; i_target < i_length; ++i_target){
                                    float i_sum = 0.0f;
                                    const size_t i_weightBase = (static_cast<size_t>(i_head) * i_length + i_target) * i_length;
                                    for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                        float i_score = 0.0f;
                                        for(uint32_t i = 0; i < i_headWidth; ++i){
                                            i_score += i_qkv[i_target * 3 * i_width + i_head * i_headWidth + i]
                                                * i_qkv[i_source * 3 * i_width + i_width + i_head * i_headWidth + i];
                                        }
                                        const float i_weight = std::exp(i_score * i_attnScale);
                                        if(!std::isfinite(i_weight)){
                                            return std::numeric_limits<float>::quiet_NaN();
                                        }
                                        i_attnWeightLayer[i_weightBase + i_source] = i_weight;
                                        i_sum += i_weight;
                                    }
                                    if(!std::isfinite(i_sum) || i_sum <= 0.0f){
                                        return std::numeric_limits<float>::quiet_NaN();
                                    }
                                    for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                        const float i_weight = i_attnWeightLayer[i_weightBase + i_source] / i_sum;
                                        i_attnWeightLayer[i_weightBase + i_source] = i_weight;
                                        for(uint32_t i = 0; i < i_headWidth; ++i){
                                            i_attention[i_target * i_width + i_head * i_headWidth + i]
                                                += i_weight * i_qkv[i_source * 3 * i_width + 2 * i_width + i_head * i_headWidth + i];
                                        }
                                    }
                                }
                            }

                            // Attention output projection and residual.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                for(uint32_t i_out = 0; i_out < i_width; ++i_out){
                                    float i_value = 0.0f;
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        i_value += i_attention[i_position * i_width + i_in]
                                            * i_attentionOutputWeights[i_attentionBase + i_out * i_width + i_in];
                                    }
                                    i_mid[i_position * i_width + i_out] =
                                        i_input[i_position * i_width + i_out] + i_value;
                                }
                            }

                            // RMS normalization before feed-forward.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                float i_mean = 0.0f;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    const float i_value = i_mid[i_position * i_width + i];
                                    i_mean += i_value * i_value;
                                }
                                const float i_rstd = 1.0f / std::sqrt(i_mean / static_cast<float>(i_width) + 1.0e-5f);
                                i_rstd2Layer[i_position] = i_rstd;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    i_normB[i_position * i_width + i] = i_mid[i_position * i_width + i]
                                        * i_rstd * i_ffnNormWeights[i_layer * i_width + i];
                                    i_output[i_position * i_width + i] = i_mid[i_position * i_width + i];
                                }
                                for(uint32_t i_out = 0; i_out < i_ffn; ++i_out){
                                    float i_gateValue = 0.0f;
                                    float i_upValue = 0.0f;
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        i_gateValue += i_normB[i_position * i_width + i_in]
                                            * i_ffnGateWeights[i_gateBase + i_out * i_width + i_in];
                                        i_upValue += i_normB[i_position * i_width + i_in]
                                            * i_ffnUpWeights[i_gateBase + i_out * i_width + i_in];
                                    }
                                    i_gateLayer[i_position * i_ffn + i_out] = i_gateValue;
                                    i_upLayer[i_position * i_ffn + i_out] = i_upValue;
                                    const float i_activated = silu(i_gateValue) * i_upValue;
                                    for(uint32_t i = 0; i < i_width; ++i){
                                        i_output[i_position * i_width + i] += i_activated
                                            * i_ffnDownWeights[i_downBase + i * i_ffn + i_out];
                                    }
                                }
                            }
                        }

                        const float* const i_final = workspaceLayer(i_layerInputStorage, i_hiddenStride, i_layers);

                        // Output projection, cross-entropy loss, and gradient at every predicted position.
                        std::vector<float>& i_logits = a_workspace.logits;
                        std::vector<float>& i_dFinal = a_workspace.dFinal;
                        zeroVectorPrefix(i_dFinal, i_hiddenSize);
                        float i_loss = 0.0f;
                        for(uint32_t i_position = 0; i_position + 1 < i_length; ++i_position){
                            const uint32_t i_targetToken = i_token[i_position + 1];
                            for(uint32_t i_v = 0; i_v < i_vocab; ++i_v){
                                float i_value = i_outputBiasWeights[i_v];
                                for(uint32_t i = 0; i < i_width; ++i){
                                    i_value += i_final[i_position * i_width + i]
                                        * i_outputWeights[i * i_vocab + i_v];
                                }
                                i_logits[i_v] = i_value;
                            }
                            const float i_maximum = *std::max_element(i_logits.begin(), i_logits.end());
                            float i_sum = 0.0f;
                            for(uint32_t i_v = 0; i_v < i_vocab; ++i_v){
                                i_logits[i_v] = std::exp(i_logits[i_v] - i_maximum);
                                i_sum += i_logits[i_v];
                            }
                            if(!std::isfinite(i_sum) || i_sum <= 0.0f){
                                return std::numeric_limits<float>::quiet_NaN();
                            }
                            for(uint32_t i_v = 0; i_v < i_vocab; ++i_v){
                                i_logits[i_v] /= i_sum;
                            }
                            i_loss -= std::log(std::max(i_logits[i_targetToken], 1.0e-9f));
                            ++a_count;

                            for(uint32_t i_v = 0; i_v < i_vocab; ++i_v){
                                const float i_delta = i_logits[i_v] - (i_v == i_targetToken ? 1.0f : 0.0f);
                                a_gradients.outputBias[i_v] += i_delta;
                                for(uint32_t i = 0; i < i_width; ++i){
                                    a_gradients.output[i * i_vocab + i_v] +=
                                        i_delta * i_final[i_position * i_width + i];
                                    i_dFinal[i_position * i_width + i] +=
                                        i_delta * i_outputWeights[i * i_vocab + i_v];
                                }
                            }
                        }

                        // Backpropagate through the transformer stack.
                        std::vector<float>& i_dHidden = a_workspace.dHidden;
                        std::copy_n(i_dFinal.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_dHidden.begin());
                        for(uint32_t i_layerReverse = 0; i_layerReverse < i_layers; ++i_layerReverse){
                            const uint32_t i_layer = i_layers - 1 - i_layerReverse;
                            const float* const i_input = workspaceLayer(i_layerInputStorage, i_hiddenStride, i_layer);
                            const float* const i_normA = workspaceLayer(i_xn1Storage, i_hiddenStride, i_layer);
                            const float* const i_normB = workspaceLayer(i_xn2Storage, i_hiddenStride, i_layer);
                            const float* const i_mid = workspaceLayer(i_hiddenMidStorage, i_hiddenStride, i_layer);
                            const float* const i_qkv = workspaceLayer(i_qkvStorage, i_qkvSize, i_layer);
                            const float* const i_attention = workspaceLayer(i_contextStorage, i_hiddenStride, i_layer);
                            const float* const i_rstd1Layer = workspaceLayer(i_rstd1Storage, i_rstdStride, i_layer);
                            const float* const i_rstd2Layer = workspaceLayer(i_rstd2Storage, i_rstdStride, i_layer);
                            const float* const i_attnWeightLayer = workspaceLayer(i_attnWeightStorage, i_attnWeightStride, i_layer);
                            const float* const i_gateLayer = workspaceLayer(i_gateStorage, i_ffnStride, i_layer);
                            const float* const i_upLayer = workspaceLayer(i_upStorage, i_ffnStride, i_layer);
                            const size_t i_qkvBase = static_cast<size_t>(i_layer) * 3 * i_width * i_width;
                            const size_t i_attentionBase = static_cast<size_t>(i_layer) * i_width * i_width;
                            const size_t i_gateBase = static_cast<size_t>(i_layer) * i_width * i_ffn;
                            const size_t i_downBase = static_cast<size_t>(i_layer) * i_ffn * i_width;

                            // Feed-forward backward. i_dHidden holds dL/d(layer output).
                            std::vector<float>& i_dNormB = a_workspace.dNormB;
                            std::vector<float>& i_dMid = a_workspace.dMid;
                            zeroVectorPrefix(i_dNormB, i_hiddenSize);
                            std::copy_n(i_dHidden.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_dMid.begin());
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                for(uint32_t i_out = 0; i_out < i_ffn; ++i_out){
                                    const float i_gateValue = i_gateLayer[i_position * i_ffn + i_out];
                                    const float i_upValue = i_upLayer[i_position * i_ffn + i_out];
                                    const float i_sig = sigmoid(i_gateValue);
                                    const float i_siluValue = i_gateValue * i_sig;
                                    float i_dActivated = 0.0f;
                                    for(uint32_t i = 0; i < i_width; ++i){
                                        i_dActivated += i_dHidden[i_position * i_width + i]
                                            * i_ffnDownWeights[i_downBase + i * i_ffn + i_out];
                                        a_gradients.ffnDown[i_downBase + i * i_ffn + i_out] +=
                                            i_dHidden[i_position * i_width + i] * (i_siluValue * i_upValue);
                                    }
                                    const float i_dSilu = i_dActivated * i_upValue;
                                    const float i_dUp = i_dActivated * i_siluValue;
                                    const float i_dGate = i_dSilu * (i_sig * (1.0f + i_gateValue * (1.0f - i_sig)));
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        a_gradients.ffnGate[i_gateBase + i_out * i_width + i_in] +=
                                            i_dGate * i_normB[i_position * i_width + i_in];
                                        a_gradients.ffnUp[i_gateBase + i_out * i_width + i_in] +=
                                            i_dUp * i_normB[i_position * i_width + i_in];
                                        i_dNormB[i_position * i_width + i_in] +=
                                            i_dGate * i_ffnGateWeights[i_gateBase + i_out * i_width + i_in]
                                            + i_dUp * i_ffnUpWeights[i_gateBase + i_out * i_width + i_in];
                                    }
                                }
                            }

                            // RMS norm (feed-forward) backward: contributes to i_dMid.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                rmsNormBackward(&i_mid[i_position * i_width],
                                                i_rstd2Layer[i_position],
                                                i_ffnNormWeights, i_layer * i_width,
                                                &i_dNormB[i_position * i_width],
                                                &i_dMid[i_position * i_width],
                                                a_gradients.ffnNorm, i_layer * i_width,
                                                i_width);
                            }

                            // Attention residual: i_dMid splits to layer input and attention output.
                            std::vector<float>& i_dInput = a_workspace.dInput;
                            std::vector<float>& i_dContext = a_workspace.dContext;
                            std::copy_n(i_dMid.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_dInput.begin());
                            zeroVectorPrefix(i_dContext, i_hiddenSize);
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                for(uint32_t i_out = 0; i_out < i_width; ++i_out){
                                    const float i_grad = i_dMid[i_position * i_width + i_out];
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        a_gradients.attentionOutput[i_attentionBase + i_out * i_width + i_in] +=
                                            i_grad * i_attention[i_position * i_width + i_in];
                                        i_dContext[i_position * i_width + i_in] +=
                                            i_grad * i_attentionOutputWeights[i_attentionBase + i_out * i_width + i_in];
                                    }
                                }
                            }

                            // Attention backward through softmax and Q/K/V.
                            std::vector<float>& i_dQkv = a_workspace.dQkv;
                            std::vector<float>& i_dWeight = a_workspace.dWeight;
                            zeroVectorPrefix(i_dQkv, i_qkvSize);
                            for(uint32_t i_head = 0; i_head < i_heads; ++i_head){
                                for(uint32_t i_target = 0; i_target < i_length; ++i_target){
                                    const size_t i_weightBase = (static_cast<size_t>(i_head) * i_length + i_target) * i_length;
                                    float i_dot = 0.0f;
                                    for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                        const float i_weight = i_attnWeightLayer[i_weightBase + i_source];
                                        float i_grad = 0.0f;
                                        for(uint32_t i = 0; i < i_headWidth; ++i){
                                            i_grad += i_dContext[i_target * i_width + i_head * i_headWidth + i]
                                                * i_qkv[i_source * 3 * i_width + 2 * i_width + i_head * i_headWidth + i];
                                            i_dQkv[i_source * 3 * i_width + 2 * i_width + i_head * i_headWidth + i] +=
                                                i_weight * i_dContext[i_target * i_width + i_head * i_headWidth + i];
                                        }
                                        i_dWeight[i_source] = i_grad;
                                        i_dot += i_weight * i_grad;
                                    }
                                    for(uint32_t i_source = 0; i_source <= i_target; ++i_source){
                                        const float i_weight = i_attnWeightLayer[i_weightBase + i_source];
                                        const float i_dScore = i_weight * (i_dWeight[i_source] - i_dot) * i_attnScale;
                                        for(uint32_t i = 0; i < i_headWidth; ++i){
                                            const float i_q = i_qkv[i_target * 3 * i_width + i_head * i_headWidth + i];
                                            const float i_k = i_qkv[i_source * 3 * i_width + i_width + i_head * i_headWidth + i];
                                            i_dQkv[i_target * 3 * i_width + i_head * i_headWidth + i] += i_dScore * i_k;
                                            i_dQkv[i_source * 3 * i_width + i_width + i_head * i_headWidth + i] += i_dScore * i_q;
                                        }
                                    }
                                }
                            }

                            // QKV projection backward.
                            std::vector<float>& i_dNormA = a_workspace.dNormA;
                            zeroVectorPrefix(i_dNormA, i_hiddenSize);
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                for(uint32_t i_out = 0; i_out < 3 * i_width; ++i_out){
                                    const float i_grad = i_dQkv[i_position * 3 * i_width + i_out];
                                    for(uint32_t i_in = 0; i_in < i_width; ++i_in){
                                        a_gradients.qkv[i_qkvBase + i_out * i_width + i_in] +=
                                            i_grad * i_normA[i_position * i_width + i_in];
                                        i_dNormA[i_position * i_width + i_in] +=
                                            i_grad * i_qkvWeights[i_qkvBase + i_out * i_width + i_in];
                                    }
                                }
                            }

                            // RMS norm (attention) backward: contributes to layer input gradient.
                            for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                                rmsNormBackward(&i_input[i_position * i_width],
                                                i_rstd1Layer[i_position],
                                                i_attentionNormWeights, i_layer * i_width,
                                                &i_dNormA[i_position * i_width],
                                                &i_dInput[i_position * i_width],
                                                a_gradients.attentionNorm, i_layer * i_width,
                                                i_width);
                            }

                            std::copy_n(i_dInput.begin(), static_cast<std::ptrdiff_t>(i_hiddenSize), i_dHidden.begin());
                        }

                        // Embedding and positional embedding gradients.
                        for(uint32_t i_position = 0; i_position < i_length; ++i_position){
                            for(uint32_t i = 0; i < i_width; ++i){
                                const float i_grad = i_dHidden[i_position * i_width + i];
                                a_gradients.embedding[i_token[i_position] * i_width + i] += i_grad;
                                a_gradients.position[i_position * i_width + i] += i_grad;
                            }
                        }

                        return i_loss;
                    }

                    // Backpropagates through RMS normalization y_i = x_i * rstd * scale_i.
                    // Accumulates dL/dx into a_dInput and dL/dscale into a_dScale.
                    static void rmsNormBackward(const float* a_input,
                                                float a_rstd,
                                                const std::vector<float>& a_scale,
                                                size_t a_scaleOffset,
                                                const float* a_dOutput,
                                                float* a_dInput,
                                                std::vector<float>& a_dScale,
                                                size_t a_dScaleOffset,
                                                uint32_t a_width){
                        float i_dot = 0.0f;
                        for(uint32_t i = 0; i < a_width; ++i){
                            i_dot += a_dOutput[i] * a_scale[a_scaleOffset + i] * a_input[i];
                        }
                        const float i_coefficient = a_rstd * a_rstd * a_rstd / static_cast<float>(a_width);
                        for(uint32_t i = 0; i < a_width; ++i){
                            a_dInput[i] += a_rstd * a_scale[a_scaleOffset + i] * a_dOutput[i]
                                - i_coefficient * a_input[i] * i_dot;
                            a_dScale[a_dScaleOffset + i] += a_dOutput[i] * a_input[i] * a_rstd;
                        }
                    }
            };


        }

    }

}
