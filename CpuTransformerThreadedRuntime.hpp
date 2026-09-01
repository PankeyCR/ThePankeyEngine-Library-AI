#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "CpuTransformer.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            class CpuTransformerThreadedRuntime{
                public:
                    explicit CpuTransformerThreadedRuntime(const CpuTransformer& a_model,
                                                            size_t a_workerCount = 0)
                        : m_model(&a_model){
                        const size_t i_workerCount = chooseWorkerCount(a_workerCount);
                        m_workspaces.resize(i_workerCount == 0 ? 1 : i_workerCount);
                        for(CpuTransformer::InferenceWorkspace& i_workspace : m_workspaces){
                            CpuTransformer::resizeInferenceWorkspace(i_workspace, a_model.m_config);
                        }
                    }

                    size_t workerCount() const{
                        return m_workspaces.size();
                    }

                    std::vector<std::vector<float>> logitsBatch(
                        const std::vector<TokenSequence>& a_batch) const{
                        std::vector<std::vector<float>> i_results(a_batch.size());
                        if(a_batch.empty()){
                            return i_results;
                        }

                        std::atomic<size_t> i_next{0};
                        const size_t i_workerCount = std::min(m_workspaces.size(), a_batch.size());
                        std::vector<std::thread> i_threads;
                        i_threads.reserve(i_workerCount);
                        for(size_t i_worker = 0; i_worker < i_workerCount; ++i_worker){
                            i_threads.emplace_back([this, &a_batch, &i_results, &i_next, i_workerCount, i_worker](){
                                const CpuTransformer::SharedParameters i_parameters = m_model->sharedParameters();
                                for(;;){
                                    const size_t i_index = i_next.fetch_add(1, std::memory_order_relaxed);
                                    if(i_index >= a_batch.size()){
                                        return;
                                    }
                                    i_results[i_index] = m_model->logits(i_parameters, m_workspaces[i_worker], a_batch[i_index]);
                                }
                            });
                        }
                        for(std::thread& i_thread : i_threads){
                            i_thread.join();
                        }
                        return i_results;
                    }

                    std::vector<std::vector<float>> probabilitiesBatch(
                        const std::vector<TokenSequence>& a_batch) const{
                        std::vector<std::vector<float>> i_results(a_batch.size());
                        if(a_batch.empty()){
                            return i_results;
                        }

                        std::atomic<size_t> i_next{0};
                        const size_t i_workerCount = std::min(m_workspaces.size(), a_batch.size());
                        std::vector<std::thread> i_threads;
                        i_threads.reserve(i_workerCount);
                        for(size_t i_worker = 0; i_worker < i_workerCount; ++i_worker){
                            i_threads.emplace_back([this, &a_batch, &i_results, &i_next, i_worker](){
                                const CpuTransformer::SharedParameters i_parameters = m_model->sharedParameters();
                                for(;;){
                                    const size_t i_index = i_next.fetch_add(1, std::memory_order_relaxed);
                                    if(i_index >= a_batch.size()){
                                        return;
                                    }
                                    i_results[i_index] = m_model->probabilities(i_parameters, m_workspaces[i_worker], a_batch[i_index]);
                                }
                            });
                        }
                        for(std::thread& i_thread : i_threads){
                            i_thread.join();
                        }
                        return i_results;
                    }

                private:
                    const CpuTransformer* m_model;
                    mutable std::vector<CpuTransformer::InferenceWorkspace> m_workspaces;

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
