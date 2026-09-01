#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include "TokenSequence.hpp"

namespace pankey{

    namespace Library{

        namespace AI{

            class TextDecoder{
                public:
                    TokenSequence encode(const char* a_text) const{
                        TokenSequence i_sequence;
                        if(a_text == nullptr){
                            return i_sequence;
                        }
                        for(const unsigned char* i_text = reinterpret_cast<const unsigned char*>(a_text);
                            *i_text != 0;
                            ++i_text){
                            i_sequence.add(static_cast<uint32_t>(*i_text));
                        }
                        return i_sequence;
                    }

                    TokenSequence loadEncode(const char* a_path) const{
                        TokenSequence i_sequence;
                        if(a_path == nullptr){
                            return i_sequence;
                        }
                        std::ifstream i_file(a_path, std::ios::binary);
                        if(!i_file){
                            return i_sequence;
                        }
                        char i_value = 0;
                        while(i_file.get(i_value)){
                            i_sequence.add(static_cast<uint32_t>(static_cast<unsigned char>(i_value)));
                        }
                        return i_sequence;
                    }

                    std::string decode(const TokenSequence& a_sequence) const{
                        std::string i_text;
                        i_text.reserve(a_sequence.size());
                        for(uint32_t i = 0; i < a_sequence.size(); ++i){
                            if(a_sequence.get(i) <= 255){
                                i_text.push_back(static_cast<char>(a_sequence.get(i)));
                            }
                        }
                        return i_text;
                    }
            };

        }

    }

}
