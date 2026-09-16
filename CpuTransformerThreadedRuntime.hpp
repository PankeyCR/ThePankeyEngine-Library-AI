#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
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
                        
                        // Create persistent worker threads
                        m_threads.reserve(m_workspaces.size());
                        for(size_t i_worker = 0; i_worker < m_workspaces.size(); ++i_worker){
                            m_threads.emplace_back(&CpuTransformerThreadedRuntime::workerLoop, this, i_worker);
                        }
                    }

                    ~CpuTransformerThreadedRuntime() noexcept{
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

                    CpuTransformerThreadedRuntime(const CpuTransformerThreadedRuntime&) = delete;
                    CpuTransformerThreadedRuntime& operator=(const CpuTransformerThreadedRuntime&) = delete;
                    CpuTransformerThreadedRuntime(CpuTransformerThreadedRuntime&&) = delete;
                    CpuTransformerThreadedRuntime& operator=(CpuTransformerThreadedRuntime&&) = delete;

                    size_t workerCount() const{
                        return m_workspaces.size();
                    }

                    std::vector<std::vector<float>> logitsBatch(
                        const std::vector<TokenSequence>& a_batch) const{
                        std::vector<std::vector<float>> i_results(a_batch.size());
                        if(a_batch.empty()){
                            return i_results;
                        }
                        executeBatch<true>(a_batch, i_results);
                        return i_results;
                    }

                    std::vector<std::vector<float>> probabilitiesBatch(
                        const std::vector<TokenSequence>& a_batch) const{
                        std::vector<std::vector<float>> i_results(a_batch.size());
                        if(a_batch.empty()){
                            return i_results;
                        }
                        executeBatch<false>(a_batch, i_results);
                        return i_results;
                    }

                private:
                    const CpuTransformer* m_model;
                    mutable std::vector<CpuTransformer::InferenceWorkspace> m_workspaces;
                    mutable std::vector<std::thread> m_threads;
                    
                    mutable std::mutex m_mutex;
                    mutable std::condition_variable m_workReady;
                    mutable std::condition_variable m_workDone;
                    
                    mutable const std::vector<TokenSequence>* m_batch = nullptr;
                    mutable std::vector<std::vector<float>>* m_results = nullptr;
                    mutable size_t m_generation = 0;
                    mutable size_t m_pendingWorkers = 0;
                    mutable bool m_isLogits = true;
                    mutable bool m_running = false;
                    mutable bool m_stopping = false;
                    mutable std::exception_ptr m_exception;

                    template<bool IsLogits>
                    void executeBatch(const std::vector<TokenSequence>& a_batch,
                                      std::vector<std::vector<float>>& a_results) const{
                        const size_t i_usefulWorkers = std::min(m_workspaces.size(), a_batch.size());
                        
                        {
                            std::lock_guard<std::mutex> i_lock(m_mutex);
                            m_batch = &a_batch;
                            m_results = &a_results;
                            m_isLogits = IsLogits;
                            m_exception = nullptr;
                            m_pendingWorkers = i_usefulWorkers;
                            m_running = true;
                            ++m_generation;
                        }
                        
                        m_workReady.notify_all();
                        
                        // Wait for all workers to complete
                        {
                            std::unique_lock<std::mutex> i_lock(m_mutex);
                            m_workDone.wait(i_lock, [this](){ return !m_running; });
                            if(m_exception){
                                std::rethrow_exception(m_exception);
                            }
                        }
                    }

                    void workerLoop(size_t a_workerIndex) noexcept{
                        size_t i_seenGeneration = 0;
                        for(;;){
                            const std::vector<TokenSequence>* i_batch = nullptr;
                            std::vector<std::vector<float>>* i_results = nullptr;
                            bool i_isLogits = true;
                            
                            {
                                std::unique_lock<std::mutex> i_lock(m_mutex);
                                m_workReady.wait(i_lock, [this, &i_seenGeneration](){
                                    return m_stopping || (m_running && m_generation != i_seenGeneration);
                                });
                                
                                if(m_stopping){
                                    return;
                                }
                                
                                i_seenGeneration = m_generation;
                                i_batch = m_batch;
                                i_results = m_results;
                                i_isLogits = m_isLogits;
                            }
                            
                            try{
                                // Static partition: worker processes its assigned range
                                const size_t i_usefulWorkers = std::min(m_workspaces.size(), i_batch->size());
                                if(a_workerIndex < i_usefulWorkers){
                                    const Range i_range = partitionRange(i_batch->size(), a_workerIndex, i_usefulWorkers);
                                    const CpuTransformer::SharedParameters i_parameters = m_model->sharedParameters();
                                    
                                    for(size_t i_index = i_range.begin; i_index < i_range.end; ++i_index){
                                        if(i_isLogits){
                                            (*i_results)[i_index] = m_model->logits(i_parameters, 
                                                                                     m_workspaces[a_workerIndex], 
                                                                                     (*i_batch)[i_index]);
                                        }else{
                                            (*i_results)[i_index] = m_model->probabilities(i_parameters, 
                                                                                           m_workspaces[a_workerIndex], 
                                                                                           (*i_batch)[i_index]);
                                        }
                                    }
                                }
                            }catch(...){
                                {
                                    std::lock_guard<std::mutex> i_lock(m_mutex);
                                    if(!m_exception){
                                        m_exception = std::current_exception();
                                    }
                                }
                            }
                            
                            // Notify main thread if all workers done
                            bool i_notify = false;
                            {
                                std::lock_guard<std::mutex> i_lock(m_mutex);
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

                    struct Range{
                        size_t begin = 0;
                        size_t end = 0;
                    };

                    static Range partitionRange(size_t a_total, size_t a_index, size_t a_parts){
                        if(a_parts == 0){
                            return Range{};
                        }
                        const size_t i_begin = (a_total * a_index) / a_parts;
                        const size_t i_end = (a_total * (a_index + 1)) / a_parts;
                        return Range{i_begin, i_end};
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
