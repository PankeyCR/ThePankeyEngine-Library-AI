#pragma once

#include <cstdint>
#include <cassert>
#include "FloatArray.hpp"

namespace pankey{

    namespace AI{

        using pankey::Type::Array::FloatArray;

        struct TensorView{
            float* m_data = nullptr;
            uint32_t m_rows = 0;
            uint32_t m_columns = 0;
            uint32_t m_stride = 0;

            bool empty() const{
                return m_data == nullptr || m_rows == 0 || m_columns == 0;
            }

            uint32_t size() const{
                return m_rows * m_columns;
            }

            float& getFast(uint32_t a_row, uint32_t a_column) const{
                assert(m_data != nullptr && "TensorView data is null");
                assert(a_row < m_rows && "Row index out of bounds");
                assert(a_column < m_columns && "Column index out of bounds");
                return m_data[a_row * m_stride + a_column];
            }

            float* row(uint32_t a_row) const{
                assert(m_data != nullptr && "TensorView data is null");
                assert(a_row < m_rows && "Row index out of bounds");
                return m_data + a_row * m_stride;
            }

            static TensorView from(FloatArray& a_buffer,
                                   uint32_t a_offset,
                                   uint32_t a_rows,
                                   uint32_t a_columns,
                                   uint32_t a_stride){
                if(a_rows == 0 || a_columns == 0 || a_stride < a_columns
                    || a_offset >= static_cast<uint32_t>(a_buffer.length())
                    || a_columns > static_cast<uint32_t>(a_buffer.length()) - a_offset
                    || a_rows - 1 > (static_cast<uint32_t>(a_buffer.length()) - a_offset - a_columns) / a_stride){
                    return TensorView{};
                }
                return TensorView{
                    a_buffer.getArrayPointer() + a_offset,
                    a_rows,
                    a_columns,
                    a_stride
                };
            }
        };

        struct ConstTensorView{
            const float* m_data = nullptr;
            uint32_t m_rows = 0;
            uint32_t m_columns = 0;
            uint32_t m_stride = 0;

            bool empty() const{
                return m_data == nullptr || m_rows == 0 || m_columns == 0;
            }

            const float& getFast(uint32_t a_row, uint32_t a_column) const{
                assert(m_data != nullptr && "ConstTensorView data is null");
                assert(a_row < m_rows && "Row index out of bounds");
                assert(a_column < m_columns && "Column index out of bounds");
                return m_data[a_row * m_stride + a_column];
            }

            static ConstTensorView from(const FloatArray& a_buffer,
                                        uint32_t a_offset,
                                        uint32_t a_rows,
                                        uint32_t a_columns,
                                        uint32_t a_stride){
                if(a_rows == 0 || a_columns == 0 || a_stride < a_columns
                    || a_offset >= static_cast<uint32_t>(a_buffer.length())
                    || a_columns > static_cast<uint32_t>(a_buffer.length()) - a_offset
                    || a_rows - 1 > (static_cast<uint32_t>(a_buffer.length()) - a_offset - a_columns) / a_stride){
                    return ConstTensorView{};
                }
                return ConstTensorView{
                    a_buffer.getArrayPointer() + a_offset,
                    a_rows,
                    a_columns,
                    a_stride
                };
            }
        };

        inline ConstTensorView asConst(const TensorView& a_view){
            return ConstTensorView{
                a_view.m_data,
                a_view.m_rows,
                a_view.m_columns,
                a_view.m_stride
            };
        }

    }

}
