#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace pankey{

    namespace Library{

        namespace AI{

            class Adam{
                public:
                    Adam(float a_learningRate = 0.001f,
                         float a_beta1 = 0.9f,
                         float a_beta2 = 0.999f,
                         float a_epsilon = 1.0e-8f)
                        : m_learningRate(a_learningRate),
                          m_beta1(a_beta1),
                          m_beta2(a_beta2),
                          m_epsilon(a_epsilon){}

                    void reset(size_t a_size){
                        m_firstMoment.assign(a_size, 0);
                        m_secondMoment.assign(a_size, 0);
                        m_step = 0;
                    }

                    bool update(std::vector<float>& a_parameters,
                                const std::vector<float>& a_gradients){
                        if(a_parameters.size() != a_gradients.size()
                            || !std::isfinite(m_learningRate)
                            || m_learningRate <= 0.0f
                            || !std::isfinite(m_beta1)
                            || m_beta1 < 0.0f || m_beta1 >= 1.0f
                            || !std::isfinite(m_beta2)
                            || m_beta2 < 0.0f || m_beta2 >= 1.0f
                            || !std::isfinite(m_epsilon)
                            || m_epsilon <= 0.0f){
                            return false;
                        }
                        if(m_firstMoment.size() != a_parameters.size()){
                            reset(a_parameters.size());
                        }
                        for(size_t i = 0; i < a_parameters.size(); ++i){
                            if(!std::isfinite(a_parameters[i]) || !std::isfinite(a_gradients[i])){
                                return false;
                            }
                        }
                        ++m_step;
                        const float i_bias1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_step));
                        const float i_bias2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_step));
                        if(!std::isfinite(i_bias1) || !std::isfinite(i_bias2)
                            || i_bias1 <= 0.0f || i_bias2 <= 0.0f){
                            return false;
                        }
                        for(size_t i = 0; i < a_parameters.size(); ++i){
                            const float i_firstMoment = decode(m_firstMoment[i]);
                            const float i_secondMoment = decode(m_secondMoment[i]);
                            if(!std::isfinite(i_firstMoment) || !std::isfinite(i_secondMoment)){
                                return false;
                            }
                            m_firstMoment[i] = encode(m_beta1 * i_firstMoment + (1.0f - m_beta1) * a_gradients[i]);
                            m_secondMoment[i] = encode(m_beta2 * i_secondMoment
                                + (1.0f - m_beta2) * a_gradients[i] * a_gradients[i]);
                            const float i_first = decode(m_firstMoment[i]) / i_bias1;
                            const float i_second = decode(m_secondMoment[i]) / i_bias2;
                            if(!std::isfinite(i_first) || !std::isfinite(i_second) || i_second < 0.0f){
                                return false;
                            }
                            const float i_update = m_learningRate * i_first / (std::sqrt(i_second) + m_epsilon);
                            if(!std::isfinite(i_update)){
                                return false;
                            }
                            a_parameters[i] -= i_update;
                            if(!std::isfinite(a_parameters[i])){
                                return false;
                            }
                        }
                        return true;
                    }

                    uint64_t step() const{
                        return m_step;
                    }

                    void setLearningRate(float a_rate){
                        m_learningRate = a_rate;
                    }

                    const std::vector<float>& firstMoment() const{
                        m_firstMomentCache.resize(m_firstMoment.size());
                        for(size_t i = 0; i < m_firstMoment.size(); ++i){
                            m_firstMomentCache[i] = decode(m_firstMoment[i]);
                        }
                        return m_firstMomentCache;
                    }

                    const std::vector<float>& secondMoment() const{
                        m_secondMomentCache.resize(m_secondMoment.size());
                        for(size_t i = 0; i < m_secondMoment.size(); ++i){
                            m_secondMomentCache[i] = decode(m_secondMoment[i]);
                        }
                        return m_secondMomentCache;
                    }

                    void loadState(uint64_t a_step,
                                   const std::vector<float>& a_first,
                                   const std::vector<float>& a_second){
                        if(a_first.size() != a_second.size()){
                            m_step = 0;
                            m_firstMoment.clear();
                            m_secondMoment.clear();
                            return;
                        }
                        for(size_t i = 0; i < a_first.size(); ++i){
                            if(!std::isfinite(a_first[i]) || !std::isfinite(a_second[i]) || a_second[i] < 0.0f){
                                m_step = 0;
                                m_firstMoment.clear();
                                m_secondMoment.clear();
                                return;
                            }
                        }
                        m_step = a_step;
                        m_firstMoment.resize(a_first.size());
                        m_secondMoment.resize(a_second.size());
                        for(size_t i = 0; i < a_first.size(); ++i){
                            m_firstMoment[i] = encode(a_first[i]);
                        }
                        for(size_t i = 0; i < a_second.size(); ++i){
                            m_secondMoment[i] = encode(a_second[i]);
                        }
                    }

                private:
                    float m_learningRate;
                    float m_beta1;
                    float m_beta2;
                    float m_epsilon;
                    uint64_t m_step = 0;
                    static uint16_t encode(float a_value){
                        union { float value; uint32_t bits; } i_value{a_value};
                        return static_cast<uint16_t>(i_value.bits >> 16);
                    }

                    static float decode(uint16_t a_value){
                        union { uint32_t bits; float value; } i_value{static_cast<uint32_t>(a_value) << 16};
                        return i_value.value;
                    }

                    std::vector<uint16_t> m_firstMoment;
                    std::vector<uint16_t> m_secondMoment;
                    mutable std::vector<float> m_firstMomentCache;
                    mutable std::vector<float> m_secondMomentCache;
            };

        }

    }

}
