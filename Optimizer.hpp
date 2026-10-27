#pragma once

#include "TrainingExample.hpp"

namespace pankey{

    namespace AI{

        enum class OptimizerType{
            SGD,
            Momentum,
            Adam
        };

        struct Optimizer{
            OptimizerType m_type = OptimizerType::SGD;
            float m_momentum = 0.9f;
            float m_beta_1 = 0.9f;
            float m_beta_2 = 0.999f;
            float m_epsilon = 0.00000001f;
            int m_step = 0;
            float m_beta_1_power = 1.0f;
            float m_beta_2_power = 1.0f;
            TensorValues m_weight_velocities;
            TensorValues m_bias_velocities;
            TensorValues m_weight_squares;
            TensorValues m_bias_squares;

            void clear(){
                m_weight_velocities.clear();
                m_bias_velocities.clear();
                m_weight_squares.clear();
                m_bias_squares.clear();
                m_step = 0;
                m_beta_1_power = 1.0f;
                m_beta_2_power = 1.0f;
            }

            bool configure(OptimizerType a_type, float a_momentum = 0.9f){
                if(a_momentum < 0.0f || a_momentum >= 1.0f){
                    return false;
                }
                m_type = a_type;
                m_momentum = a_momentum;
                clear();
                return true;
            }

            bool configureAdam(float a_beta_1 = 0.9f, float a_beta_2 = 0.999f, float a_epsilon = 0.00000001f){
                if(a_beta_1 < 0.0f || a_beta_1 >= 1.0f
                    || a_beta_2 < 0.0f || a_beta_2 >= 1.0f
                    || a_epsilon <= 0.0f){
                    return false;
                }
                m_type = OptimizerType::Adam;
                m_beta_1 = a_beta_1;
                m_beta_2 = a_beta_2;
                m_epsilon = a_epsilon;
                clear();
                return true;
            }

            bool apply(
                TensorValues& a_weights,
                TensorValues& a_biases,
                const GradientBuffer& a_gradients,
                int a_first_trainable_bias,
                float a_learning_rate,
                int a_example_count
            ){
                if(a_learning_rate <= 0.0f
                    || a_example_count <= 0
                    || a_first_trainable_bias < 0
                    || a_first_trainable_bias > a_biases.length()
                    || a_gradients.m_weight_gradients.length() != a_weights.length()
                    || a_gradients.m_bias_gradients.length() != a_biases.length()
                    || !prepare(a_weights.length(), a_biases.length())){
                    return false;
                }

                const float i_scale = 1.0f / static_cast<float>(a_example_count);
                if(m_type == OptimizerType::Adam){
                    ++m_step;
                    m_beta_1_power *= m_beta_1;
                    m_beta_2_power *= m_beta_2;
                }
                for(int i = 0; i < a_weights.length(); ++i){
                    const float i_gradient = a_gradients.m_weight_gradients.getFast(i) * i_scale;
                    a_weights.setFast(i, a_weights.getFast(i) - a_learning_rate * updateWeight(i, i_gradient));
                }
                for(int i = a_first_trainable_bias; i < a_biases.length(); ++i){
                    const float i_gradient = a_gradients.m_bias_gradients.getFast(i) * i_scale;
                    a_biases.setFast(i, a_biases.getFast(i) - a_learning_rate * updateBias(i, i_gradient));
                }

                return true;
            }

        private:
            bool prepare(int a_weight_count, int a_bias_count){
                const bool i_first_moment_ready = m_weight_velocities.length() == a_weight_count
                    && m_bias_velocities.length() == a_bias_count;
                const bool i_second_moment_ready = m_weight_squares.length() == a_weight_count
                    && m_bias_squares.length() == a_bias_count;
                if(i_first_moment_ready && (m_type != OptimizerType::Adam || i_second_moment_ready)){
                    return true;
                }

                clear();
                m_weight_velocities.createArray(a_weight_count);
                m_weight_velocities.setLength(a_weight_count);
                m_bias_velocities.createArray(a_bias_count);
                m_bias_velocities.setLength(a_bias_count);
                for(int i = 0; i < a_weight_count; ++i){
                    m_weight_velocities.setFast(i, 0.0f);
                }
                for(int i = 0; i < a_bias_count; ++i){
                    m_bias_velocities.setFast(i, 0.0f);
                }

                if(m_type == OptimizerType::Adam){
                    m_weight_squares.createArray(a_weight_count);
                    m_weight_squares.setLength(a_weight_count);
                    m_bias_squares.createArray(a_bias_count);
                    m_bias_squares.setLength(a_bias_count);
                    for(int i = 0; i < a_weight_count; ++i){
                        m_weight_squares.setFast(i, 0.0f);
                    }
                    for(int i = 0; i < a_bias_count; ++i){
                        m_bias_squares.setFast(i, 0.0f);
                    }
                }
                return true;
            }

            float updateWeight(int a_index, float a_gradient){
                if(m_type == OptimizerType::SGD){
                    return a_gradient;
                }
                if(m_type == OptimizerType::Adam){
                    const float i_first = m_beta_1 * m_weight_velocities.getFast(a_index) + (1.0f - m_beta_1) * a_gradient;
                    const float i_second = m_beta_2 * m_weight_squares.getFast(a_index) + (1.0f - m_beta_2) * a_gradient * a_gradient;
                    m_weight_velocities.setFast(a_index, i_first);
                    m_weight_squares.setFast(a_index, i_second);
                    return (i_first / (1.0f - m_beta_1_power))
                        / (squareRoot(i_second / (1.0f - m_beta_2_power)) + m_epsilon);
                }
                const float i_velocity = m_momentum * m_weight_velocities.getFast(a_index) + a_gradient;
                m_weight_velocities.setFast(a_index, i_velocity);
                return i_velocity;
            }

            float updateBias(int a_index, float a_gradient){
                if(m_type == OptimizerType::SGD){
                    return a_gradient;
                }
                if(m_type == OptimizerType::Adam){
                    const float i_first = m_beta_1 * m_bias_velocities.getFast(a_index) + (1.0f - m_beta_1) * a_gradient;
                    const float i_second = m_beta_2 * m_bias_squares.getFast(a_index) + (1.0f - m_beta_2) * a_gradient * a_gradient;
                    m_bias_velocities.setFast(a_index, i_first);
                    m_bias_squares.setFast(a_index, i_second);
                    return (i_first / (1.0f - m_beta_1_power))
                        / (squareRoot(i_second / (1.0f - m_beta_2_power)) + m_epsilon);
                }
                const float i_velocity = m_momentum * m_bias_velocities.getFast(a_index) + a_gradient;
                m_bias_velocities.setFast(a_index, i_velocity);
                return i_velocity;
            }

            static float squareRoot(float a_value){
                if(a_value <= 0.0f){
                    return 0.0f;
                }

                float i_estimate = a_value > 1.0f ? a_value : 1.0f;
                for(int i = 0; i < 8; ++i){
                    i_estimate = 0.5f * (i_estimate + a_value / i_estimate);
                }
                return i_estimate;
            }
        };
    }
}