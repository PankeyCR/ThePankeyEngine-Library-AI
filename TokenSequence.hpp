#pragma once

#include <cstdint>
#include <vector>

namespace pankey{

    namespace Library{

        namespace AI{

            struct TokenSequence{
                std::vector<uint32_t> m_tokens;

                bool empty() const{
                    return m_tokens.empty();
                }

                uint32_t size() const{
                    return static_cast<uint32_t>(m_tokens.size());
                }

                void add(uint32_t a_token){
                    m_tokens.push_back(a_token);
                }

                uint32_t get(uint32_t a_index) const{
                    return m_tokens[a_index];
                }
            };

        }

    }

}
