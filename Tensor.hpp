#pragma once

#include "ArrayPointer.hpp"
#include "MemoryAllocator.hpp"

namespace pankey{

    namespace AI{

        struct TensorSizeModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = int;
            using SIZE_TYPE = int;
        };

        struct TensorValueModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = float;
            using SIZE_TYPE = int;
        };

        using TensorSizes = pankey::DataStructure::Array::ArrayPointer<TensorSizeModel>;
        using TensorValues = pankey::DataStructure::Array::ArrayPointer<TensorValueModel>;

        template<class... Args>
        TensorSizes createTensorSizes(Args... a_args){
            return pankey::DataStructure::Array::createArrayPointer<TensorSizeModel>(a_args...);
        }

        struct Tensor{

            TensorSizes m_shape;
            TensorSizes m_strides;
            TensorValues m_data;

            Tensor() = default;

            Tensor(const TensorSizes& a_shape){
                create(a_shape);
            }

            bool create(const TensorSizes& a_shape){
                if(a_shape.isEmpty()){
                    clear();
                    return false;
                }

                int i_element_count = 1;

                for(int i = 0; i < a_shape.length(); ++i){
                    if(a_shape.getFast(i) <= 0){
                        clear();
                        return false;
                    }
                    i_element_count *= a_shape.getFast(i);
                }

                m_shape.createArray(a_shape.length());
                m_strides.createArray(a_shape.length());
                m_data.createArray(i_element_count);

                for(int i = 0; i < a_shape.length(); ++i){
                    m_shape.setFast(i, a_shape.getFast(i));
                }
                m_shape.setLength(a_shape.length());

                m_strides.setFast(a_shape.length() - 1, 1);
                for(int i = a_shape.length() - 1; i > 0; --i){
                    m_strides.setFast(i - 1, m_strides.getFast(i) * m_shape.getFast(i));
                }
                m_strides.setLength(a_shape.length());

                for(int i = 0; i < i_element_count; ++i){
                    m_data.setFast(i, 0.0f);
                }
                m_data.setLength(i_element_count);
                return true;
            }

            void clear(){
                m_shape.clear();
                m_strides.clear();
                m_data.clear();
            }

            int rank() const{
                return m_shape.length();
            }

            int size() const{
                return m_data.length();
            }

            bool empty() const{
                return m_data.isEmpty();
            }

            const TensorSizes& shape() const{
                return m_shape;
            }

            bool set(const TensorSizes& a_indices, float a_value){
                int i_offset = 0;
                if(!offset(a_indices, i_offset)){
                    return false;
                }
                m_data.setFast(i_offset, a_value);
                return true;
            }

            bool get(const TensorSizes& a_indices, float& a_value) const{
                int i_offset = 0;
                if(!offset(a_indices, i_offset)){
                    return false;
                }
                a_value = m_data.getFast(i_offset);
                return true;
            }

            const TensorValues& values() const{
                return m_data;
            }

        private:
            bool offset(const TensorSizes& a_indices, int& a_offset) const{
                if(a_indices.length() != m_shape.length()){
                    return false;
                }

                a_offset = 0;
                for(int i = 0; i < a_indices.length(); ++i){
                    if(a_indices.getFast(i) < 0 || a_indices.getFast(i) >= m_shape.getFast(i)){
                        return false;
                    }
                    a_offset += a_indices.getFast(i) * m_strides.getFast(i);
                }
                return true;
            }
        };
    }
}