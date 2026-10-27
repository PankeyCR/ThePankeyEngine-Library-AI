#pragma once

#include "GGUFFile.hpp"

namespace pankey{

    namespace AI{

        struct GGUFByteModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = unsigned char;
            using SIZE_TYPE = int;
        };

        using GGUFBytes = pankey::DataStructure::Array::ArrayPointer<GGUFByteModel>;

        struct GGUFModelLoader{
            pankey::FileSystem::GGUFModel m_model;

            bool load(pankey::FileSystem::BinaryFile& a_file){
                return pankey::FileSystem::GGUFFile::load(a_file, m_model);
            }

            void clear(){
                m_model.clear();
            }

            int tensorCount() const{
                return m_model.m_tensors.length();
            }

            bool tensor(int a_index, pankey::FileSystem::GGUFTensor& a_tensor) const{
                if(a_index < 0 || a_index >= m_model.m_tensors.length()){
                    return false;
                }
                a_tensor = m_model.m_tensors.getFast(a_index);
                return true;
            }

            bool seekTensor(
                pankey::FileSystem::BinaryFile& a_file,
                int a_index
            ) const{
                pankey::FileSystem::GGUFTensor i_tensor;
                if(!tensor(a_index, i_tensor)
                    || i_tensor.m_offset > 18446744073709551615ull - m_model.m_tensor_data_offset){
                    return false;
                }
                return a_file.seek(m_model.m_tensor_data_offset + i_tensor.m_offset);
            }

            bool readTensorBytes(
                pankey::FileSystem::BinaryFile& a_file,
                int a_index,
                int a_byte_count,
                GGUFBytes& a_bytes
            ) const{
                if(a_byte_count <= 0 || !seekTensor(a_file, a_index)){
                    return false;
                }

                a_bytes.createArray(a_byte_count);
                return a_file.readArray(a_bytes, a_byte_count);
            }
        };
    }
}