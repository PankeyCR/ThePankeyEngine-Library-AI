#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "CpuTransformer.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            class CpuTransformerThreadedTrainer{
                public:
                    explicit CpuTransformerThreadedTrainer(size_t a_workerCount = 0)
                        : m_workerCount(chooseWorkerCount(a_workerCount)){
                        m_workers.resize(m_workerCount);
                        m_partialNormSquares.assign(m_workerCount, 0.0);
                        m_updateAssignments.assign(m_workerCount, std::vector<size_t>());
                        for(size_t i_worker = 0; i_worker < m_workerCount; ++i_worker){
                            m_threads.emplace_back(&CpuTransformerThreadedTrainer::workerLoop, this, i_worker);
                        }
                    }

                    ~CpuTransformerThreadedTrainer(){
                        stopWorkers();
                    }

                    CpuTransformerThreadedTrainer(const CpuTransformerThreadedTrainer&) = delete;
                    CpuTransformerThreadedTrainer& operator=(const CpuTransformerThreadedTrainer&) = delete;
                    CpuTransformerThreadedTrainer(CpuTransformerThreadedTrainer&&) = delete;
                    CpuTransformerThreadedTrainer& operator=(CpuTransformerThreadedTrainer&&) = delete;

                    size_t workerCount() const{
                        return m_workerCount;
                    }

                    float train(CpuTransformer& a_model,
                                const std::vector<TokenSequence>& a_batch,
                                float a_learningRate){
                        if(a_batch.empty() || m_workerCount == 0){
                            return 0.0f;
                        }
                        if(!std::isfinite(a_learningRate) || a_learningRate <= 0.0f){
                            throw std::invalid_argument("invalid transformer learning rate");
                        }
                        if(!a_model.parametersFinite()){
                            throw std::runtime_error("transformer parameters must stay finite");
                        }
                        ensureWorkers(a_model);
                        {
                            std::unique_lock<std::mutex> i_lock(m_mutex);
                            if(m_running){
                                throw std::logic_error("transformer trainer train already in progress");
                            }
                            m_losses.assign(m_workers.size(), 0.0f);
                            m_counts.assign(m_workers.size(), 0);
                            m_batch = &a_batch;
                            m_model = &a_model;
                            m_parameters = a_model.sharedParameters();
                        }
                        {
                            float i_loss = 0.0f;
                            uint32_t i_count = 0;
                            float i_result = 0.0f;
                            try{
                                runPhase(Phase::AccumulateBatch);

                                for(size_t i_worker = 0; i_worker < m_workers.size(); ++i_worker){
                                    i_loss += m_losses[i_worker];
                                    i_count += m_counts[i_worker];
                                }
                                if(i_count == 0){
                                    CpuTransformer::zeroGradients(a_model.m_gradientWorkspace);
                                }else{
                                    const float i_scale = 1.0f / static_cast<float>(i_count);
                                    m_gradientScale = i_scale;
                                    runPhase(Phase::ReduceScaleAndAccumulateNorm);

                                    double i_sumSquares = 0.0;
                                    for(double i_partial : m_partialNormSquares){
                                        i_sumSquares += i_partial;
                                    }
                                    if(!std::isfinite(i_sumSquares) || i_sumSquares < 0.0){
                                        throw std::runtime_error("non-finite transformer gradients");
                                    }
                                    const double i_norm = std::sqrt(i_sumSquares);
                                    if(!std::isfinite(i_norm)){
                                        throw std::runtime_error("non-finite transformer gradients");
                                    }
                                    if(i_norm > 1.0){
                                        m_gradientScale = 1.0f / static_cast<float>(i_norm);
                                        runPhase(Phase::ScaleGradients);
                                    }

                                    setLearningRate(a_model, a_learningRate);
                                    runPhase(Phase::UpdateParameters);
                                    if(!a_model.parametersFinite()){
                                        throw std::runtime_error("non-finite transformer parameters after update");
                                    }

                                    const float i_meanLoss = i_loss * i_scale;
                                    if(!std::isfinite(i_meanLoss)){
                                        throw std::runtime_error("non-finite transformer mean loss");
                                    }
                                    i_result = i_meanLoss;
                                }
                            }catch(...){
                                clearTrainState();
                                throw;
                            }
                            clearTrainState();
                            return i_result;
                        }
                    }

                private:
                    struct WorkerState{
                        CpuTransformer::TrainGradients gradients;
                        CpuTransformer::TrainingWorkspace workspace;
                    };

                    enum class Phase{
                        Idle,
                        AccumulateBatch,
                        ReduceScaleAndAccumulateNorm,
                        ScaleGradients,
                        UpdateParameters
                    };

                    struct Range{
                        size_t begin = 0;
                        size_t end = 0;
                    };

                    size_t m_workerCount;
                    CpuTransformer* m_model = nullptr;
                    std::vector<WorkerState> m_workers;
                    std::vector<std::thread> m_threads;
                    std::vector<float> m_losses;
                    std::vector<uint32_t> m_counts;
                    std::vector<double> m_partialNormSquares;
                    std::vector<std::vector<size_t>> m_updateAssignments;
                    const std::vector<TokenSequence>* m_batch = nullptr;
                    CpuTransformer::SharedParameters m_parameters;
                    CpuTransformer::ParameterSizes m_parameterSizes;
                    std::mutex m_mutex;
                    std::condition_variable m_workReady;
                    std::condition_variable m_workDone;
                    size_t m_generation = 0;
                    size_t m_pendingWorkers = 0;
                    bool m_running = false;
                    bool m_stopping = false;
                    bool m_hasParameterSizes = false;
                    float m_gradientScale = 1.0f;
                    Phase m_phase = Phase::Idle;
                    std::exception_ptr m_exception;

                    void ensureWorkers(const CpuTransformer& a_model){
                        std::unique_lock<std::mutex> i_lock(m_mutex);
                        if(m_running){
                            throw std::logic_error("transformer trainer train already in progress");
                        }
                        const CpuTransformer::ParameterSizes i_sizes = CpuTransformer::expectedParameterSizes(a_model.m_config);
                        if(m_hasParameterSizes && sameParameterSizes(i_sizes, m_parameterSizes)){
                            return;
                        }
                        for(WorkerState& i_worker : m_workers){
                            CpuTransformer::resizeGradientWorkspace(i_worker.gradients, i_sizes);
                            CpuTransformer::resizeTrainingWorkspace(i_worker.workspace, a_model.m_config);
                        }
                        m_parameterSizes = i_sizes;
                        m_hasParameterSizes = true;
                        buildUpdateAssignments(i_sizes);
                    }

                    void stopWorkers() noexcept{
                        {
                            std::lock_guard<std::mutex> i_lock(m_mutex);
                            m_stopping = true;
                        }
                        m_workReady.notify_all();
                        for(std::thread& i_thread : m_threads){
                            if(i_thread.joinable()){
                                i_thread.join();
                            }
                        }
                    }

                    void workerLoop(size_t a_workerIndex) noexcept{
                        size_t i_seenGeneration = 0;
                        for(;;){
                            CpuTransformer* i_model = nullptr;
                            const std::vector<TokenSequence>* i_batch = nullptr;
                            CpuTransformer::SharedParameters i_parameters;
                            Phase i_phase = Phase::Idle;
                            float i_gradientScale = 1.0f;
                            {
                                std::unique_lock<std::mutex> i_lock(m_mutex);
                                m_workReady.wait(i_lock, [this, &i_seenGeneration](){
                                    return m_stopping || (m_running && m_generation != i_seenGeneration);
                                });
                                if(m_stopping){
                                    return;
                                }
                                i_seenGeneration = m_generation;
                                i_model = m_model;
                                i_batch = m_batch;
                                i_parameters = m_parameters;
                                i_phase = m_phase;
                                i_gradientScale = m_gradientScale;
                            }

                            WorkerState& i_state = m_workers[a_workerIndex];
                            float i_loss = 0.0f;
                            uint32_t i_count = 0;
                            double i_normSquare = 0.0;
                            std::exception_ptr i_exception;
                            try{
                                runWorkerPhase(a_workerIndex,
                                               i_phase,
                                               *i_model,
                                               i_batch,
                                               i_parameters,
                                               i_state,
                                               i_loss,
                                               i_count,
                                               i_normSquare,
                                               i_gradientScale);
                            }catch(...){
                                if(i_phase == Phase::AccumulateBatch){
                                    CpuTransformer::zeroGradients(i_state.gradients);
                                }
                                i_loss = 0.0f;
                                i_count = 0;
                                i_normSquare = 0.0;
                                i_exception = std::current_exception();
                            }

                            m_losses[a_workerIndex] = i_loss;
                            m_counts[a_workerIndex] = i_count;
                            m_partialNormSquares[a_workerIndex] = i_normSquare;

                            bool i_notify = false;
                            {
                                std::lock_guard<std::mutex> i_lock(m_mutex);
                                if(i_exception != nullptr && m_exception == nullptr){
                                    m_exception = i_exception;
                                }
                                if(m_pendingWorkers > 0){
                                    --m_pendingWorkers;
                                }
                                if(m_pendingWorkers == 0){
                                    m_running = false;
                                    i_notify = true;
                                }
                            }
                            if(i_notify){
                                m_workDone.notify_one();
                            }
                        }
                    }

                    void clearTrainState(){
                        std::unique_lock<std::mutex> i_lock(m_mutex);
                        m_batch = nullptr;
                        m_model = nullptr;
                        m_phase = Phase::Idle;
                        m_exception = nullptr;
                    }

                    void runPhase(Phase a_phase){
                        {
                            std::unique_lock<std::mutex> i_lock(m_mutex);
                            if(m_running){
                                throw std::logic_error("transformer trainer train already in progress");
                            }
                            m_phase = a_phase;
                            m_exception = nullptr;
                            m_pendingWorkers = m_workers.size();
                            m_running = true;
                            ++m_generation;
                        }
                        m_workReady.notify_all();

                        std::exception_ptr i_exception;
                        {
                            std::unique_lock<std::mutex> i_lock(m_mutex);
                            m_workDone.wait(i_lock, [this](){
                                return !m_running;
                            });
                            i_exception = m_exception;
                        }
                        if(i_exception != nullptr){
                            std::rethrow_exception(i_exception);
                        }
                    }

                    static bool sameParameterSizes(const CpuTransformer::ParameterSizes& a_left,
                                                   const CpuTransformer::ParameterSizes& a_right){
                        return a_left.embedding == a_right.embedding
                            && a_left.position == a_right.position
                            && a_left.output == a_right.output
                            && a_left.outputBias == a_right.outputBias
                            && a_left.qkv == a_right.qkv
                            && a_left.attentionOutput == a_right.attentionOutput
                            && a_left.attentionNorm == a_right.attentionNorm
                            && a_left.ffnGate == a_right.ffnGate
                            && a_left.ffnUp == a_right.ffnUp
                            && a_left.ffnDown == a_right.ffnDown
                            && a_left.ffnNorm == a_right.ffnNorm;
                    }

                    static constexpr size_t gradientTensorCount(){
                        return 11;
                    }

                    static std::vector<float>& gradientTensor(CpuTransformer::TrainGradients& a_gradients,
                                                              size_t a_index){
                        switch(a_index){
                            case 0: return a_gradients.embedding;
                            case 1: return a_gradients.position;
                            case 2: return a_gradients.output;
                            case 3: return a_gradients.outputBias;
                            case 4: return a_gradients.qkv;
                            case 5: return a_gradients.attentionOutput;
                            case 6: return a_gradients.attentionNorm;
                            case 7: return a_gradients.ffnGate;
                            case 8: return a_gradients.ffnUp;
                            case 9: return a_gradients.ffnDown;
                            default: return a_gradients.ffnNorm;
                        }
                    }

                    static const std::vector<float>& gradientTensor(const CpuTransformer::TrainGradients& a_gradients,
                                                                    size_t a_index){
                        switch(a_index){
                            case 0: return a_gradients.embedding;
                            case 1: return a_gradients.position;
                            case 2: return a_gradients.output;
                            case 3: return a_gradients.outputBias;
                            case 4: return a_gradients.qkv;
                            case 5: return a_gradients.attentionOutput;
                            case 6: return a_gradients.attentionNorm;
                            case 7: return a_gradients.ffnGate;
                            case 8: return a_gradients.ffnUp;
                            case 9: return a_gradients.ffnDown;
                            default: return a_gradients.ffnNorm;
                        }
                    }

                    static size_t parameterGroupSize(const CpuTransformer::ParameterSizes& a_sizes,
                                                     size_t a_groupIndex){
                        switch(a_groupIndex){
                            case 0: return a_sizes.embedding;
                            case 1: return a_sizes.position;
                            case 2: return a_sizes.output;
                            case 3: return a_sizes.outputBias;
                            case 4: return a_sizes.qkv;
                            case 5: return a_sizes.attentionOutput;
                            case 6: return a_sizes.attentionNorm;
                            case 7: return a_sizes.ffnGate;
                            case 8: return a_sizes.ffnUp;
                            case 9: return a_sizes.ffnDown;
                            default: return a_sizes.ffnNorm;
                        }
                    }

                    static Range partitionRange(size_t a_total,
                                                size_t a_index,
                                                size_t a_parts){
                        if(a_parts == 0){
                            return Range{};
                        }
                        const size_t i_begin = (a_total * a_index) / a_parts;
                        const size_t i_end = (a_total * (a_index + 1)) / a_parts;
                        return Range{i_begin, i_end};
                    }

                    void buildUpdateAssignments(const CpuTransformer::ParameterSizes& a_sizes){
                        m_updateAssignments.assign(m_workerCount, std::vector<size_t>());
                        if(m_workerCount == 0){
                            return;
                        }
                        const std::array<size_t, gradientTensorCount()> i_groupSizes = {
                            a_sizes.embedding,
                            a_sizes.position,
                            a_sizes.output,
                            a_sizes.outputBias,
                            a_sizes.qkv,
                            a_sizes.attentionOutput,
                            a_sizes.attentionNorm,
                            a_sizes.ffnGate,
                            a_sizes.ffnUp,
                            a_sizes.ffnDown,
                            a_sizes.ffnNorm
                        };
                        std::array<size_t, gradientTensorCount()> i_groupIndices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                        std::sort(i_groupIndices.begin(),
                                  i_groupIndices.end(),
                                  [&i_groupSizes](size_t a_left, size_t a_right){
                                      return i_groupSizes[a_left] > i_groupSizes[a_right];
                                  });
                        std::vector<size_t> i_loads(m_workerCount, 0);
                        for(size_t i_group : i_groupIndices){
                            size_t i_bestWorker = 0;
                            for(size_t i_worker = 1; i_worker < m_workerCount; ++i_worker){
                                if(i_loads[i_worker] < i_loads[i_bestWorker]){
                                    i_bestWorker = i_worker;
                                }
                            }
                            m_updateAssignments[i_bestWorker].push_back(i_group);
                            i_loads[i_bestWorker] += i_groupSizes[i_group];
                        }
                    }

                    size_t totalGradientScalars() const{
                        if(!m_hasParameterSizes){
                            return 0;
                        }
                        size_t i_total = 0;
                        for(size_t i_group = 0; i_group < gradientTensorCount(); ++i_group){
                            i_total += parameterGroupSize(m_parameterSizes, i_group);
                        }
                        return i_total;
                    }

                    void runWorkerPhase(size_t a_workerIndex,
                                        Phase a_phase,
                                        CpuTransformer& a_model,
                                        const std::vector<TokenSequence>* a_batch,
                                        const CpuTransformer::SharedParameters& a_parameters,
                                        WorkerState& a_state,
                                        float& a_loss,
                                        uint32_t& a_count,
                                        double& a_normSquare,
                                        float a_gradientScale){
                        switch(a_phase){
                            case Phase::AccumulateBatch:
                                runAccumulatePhase(a_workerIndex,
                                                   a_model,
                                                   *a_batch,
                                                   a_parameters,
                                                   a_state,
                                                   a_loss,
                                                   a_count);
                                break;
                            case Phase::ReduceScaleAndAccumulateNorm:
                                runReduceScaleAndAccumulateNormPhase(a_workerIndex,
                                                                     a_model,
                                                                     a_normSquare,
                                                                     a_gradientScale);
                                break;
                            case Phase::ScaleGradients:
                                runScaleGradientPhase(a_workerIndex,
                                                      a_model.m_gradientWorkspace,
                                                      a_gradientScale);
                                break;
                            case Phase::UpdateParameters:
                                runUpdatePhase(a_workerIndex, a_model);
                                break;
                            case Phase::Idle:
                            default:
                                break;
                        }
                    }

                    void runAccumulatePhase(size_t a_workerIndex,
                                            const CpuTransformer& a_model,
                                            const std::vector<TokenSequence>& a_batch,
                                            const CpuTransformer::SharedParameters& a_parameters,
                                            WorkerState& a_state,
                                            float& a_loss,
                                            uint32_t& a_count){
                        CpuTransformer::zeroGradients(a_state.gradients);
                        const Range i_range = partitionRange(a_batch.size(), a_workerIndex, m_workers.size());
                        for(size_t i_index = i_range.begin; i_index < i_range.end; ++i_index){
                            const float i_sequenceLoss = a_model.accumulateSequence(a_parameters,
                                                                                    a_state.workspace,
                                                                                    a_batch[i_index],
                                                                                    a_state.gradients,
                                                                                    a_count);
                            if(!std::isfinite(i_sequenceLoss)){
                                throw std::runtime_error("non-finite transformer loss");
                            }
                            a_loss += i_sequenceLoss;
                        }
                    }

                    void runReduceScaleAndAccumulateNormPhase(size_t a_workerIndex,
                                                              CpuTransformer& a_model,
                                                              double& a_normSquare,
                                                              float a_scale){
                        const Range i_range = partitionRange(totalGradientScalars(), a_workerIndex, m_workers.size());
                        if(i_range.begin == i_range.end){
                            return;
                        }
                        size_t i_globalOffset = 0;
                        for(size_t i_group = 0; i_group < gradientTensorCount(); ++i_group){
                            std::vector<float>& i_target = gradientTensor(a_model.m_gradientWorkspace, i_group);
                            const size_t i_groupSize = i_target.size();
                            const size_t i_groupEnd = i_globalOffset + i_groupSize;
                            const size_t i_begin = std::max(i_range.begin, i_globalOffset);
                            const size_t i_end = std::min(i_range.end, i_groupEnd);
                            if(i_begin < i_end){
                                const size_t i_localBegin = i_begin - i_globalOffset;
                                const size_t i_localEnd = i_end - i_globalOffset;
                                for(size_t i_index = i_localBegin; i_index < i_localEnd; ++i_index){
                                    float i_value = 0.0f;
                                    for(const WorkerState& i_worker : m_workers){
                                        i_value += gradientTensor(i_worker.gradients, i_group)[i_index];
                                    }
                                    i_value *= a_scale;
                                    if(!std::isfinite(i_value)){
                                        throw std::runtime_error("non-finite transformer gradients");
                                    }
                                    i_target[i_index] = i_value;
                                    a_normSquare += static_cast<double>(i_value) * static_cast<double>(i_value);
                                }
                            }
                            i_globalOffset = i_groupEnd;
                            if(i_globalOffset >= i_range.end){
                                break;
                            }
                        }
                    }

                    void runScaleGradientPhase(size_t a_workerIndex,
                                               CpuTransformer::TrainGradients& a_gradients,
                                               float a_scale){
                        const Range i_range = partitionRange(totalGradientScalars(), a_workerIndex, m_workers.size());
                        if(i_range.begin == i_range.end){
                            return;
                        }
                        size_t i_globalOffset = 0;
                        for(size_t i_group = 0; i_group < gradientTensorCount(); ++i_group){
                            std::vector<float>& i_values = gradientTensor(a_gradients, i_group);
                            const size_t i_groupSize = i_values.size();
                            const size_t i_groupEnd = i_globalOffset + i_groupSize;
                            const size_t i_begin = std::max(i_range.begin, i_globalOffset);
                            const size_t i_end = std::min(i_range.end, i_groupEnd);
                            if(i_begin < i_end){
                                const size_t i_localBegin = i_begin - i_globalOffset;
                                const size_t i_localEnd = i_end - i_globalOffset;
                                for(size_t i_index = i_localBegin; i_index < i_localEnd; ++i_index){
                                    i_values[i_index] *= a_scale;
                                }
                            }
                            i_globalOffset = i_groupEnd;
                            if(i_globalOffset >= i_range.end){
                                break;
                            }
                        }
                    }

                    void runUpdatePhase(size_t a_workerIndex,
                                        CpuTransformer& a_model){
                        for(size_t i_group : m_updateAssignments[a_workerIndex]){
                            if(!updateParameterGroup(a_model, i_group)){
                                throw std::runtime_error("transformer optimizer update failed");
                            }
                        }
                    }

                    static void setLearningRate(CpuTransformer& a_model,
                                                float a_learningRate){
                        a_model.m_optEmbedding.setLearningRate(a_learningRate);
                        a_model.m_optPosition.setLearningRate(a_learningRate);
                        a_model.m_optOutput.setLearningRate(a_learningRate);
                        a_model.m_optOutputBias.setLearningRate(a_learningRate);
                        a_model.m_optQkv.setLearningRate(a_learningRate);
                        a_model.m_optAttentionOutput.setLearningRate(a_learningRate);
                        a_model.m_optAttentionNorm.setLearningRate(a_learningRate);
                        a_model.m_optFfnGate.setLearningRate(a_learningRate);
                        a_model.m_optFfnUp.setLearningRate(a_learningRate);
                        a_model.m_optFfnDown.setLearningRate(a_learningRate);
                        a_model.m_optFfnNorm.setLearningRate(a_learningRate);
                    }

                    static bool updateParameterGroup(CpuTransformer& a_model,
                                                     size_t a_groupIndex){
                        switch(a_groupIndex){
                            case 0:
                                return a_model.m_optEmbedding.update(a_model.m_embedding, a_model.m_gradientWorkspace.embedding);
                            case 1:
                                return a_model.m_optPosition.update(a_model.m_position, a_model.m_gradientWorkspace.position);
                            case 2:
                                return a_model.m_optOutput.update(a_model.m_output, a_model.m_gradientWorkspace.output);
                            case 3:
                                return a_model.m_optOutputBias.update(a_model.m_outputBias, a_model.m_gradientWorkspace.outputBias);
                            case 4:
                                return a_model.m_optQkv.update(a_model.m_qkv, a_model.m_gradientWorkspace.qkv);
                            case 5:
                                return a_model.m_optAttentionOutput.update(a_model.m_attentionOutput, a_model.m_gradientWorkspace.attentionOutput);
                            case 6:
                                return a_model.m_optAttentionNorm.update(a_model.m_attentionNorm, a_model.m_gradientWorkspace.attentionNorm);
                            case 7:
                                return a_model.m_optFfnGate.update(a_model.m_ffnGate, a_model.m_gradientWorkspace.ffnGate);
                            case 8:
                                return a_model.m_optFfnUp.update(a_model.m_ffnUp, a_model.m_gradientWorkspace.ffnUp);
                            case 9:
                                return a_model.m_optFfnDown.update(a_model.m_ffnDown, a_model.m_gradientWorkspace.ffnDown);
                            default:
                                return a_model.m_optFfnNorm.update(a_model.m_ffnNorm, a_model.m_gradientWorkspace.ffnNorm);
                        }
                    }

                    static size_t chooseWorkerCount(size_t a_requested){
                        if(a_requested != 0){
                            return a_requested;
                        }
                        const size_t i_hardware = std::thread::hardware_concurrency();
                        return i_hardware == 0 ? 1 : i_hardware;
                    }
            };

        }

    }

}
