
#ifndef Network_hpp
	#define Network_hpp

	#include "ArrayPointer.hpp"

	#if defined(pankey_Log) && (defined(Network_Log) || defined(pankey_Global_Log) || defined(pankey_Base_Log))
		#include "Logger_status.hpp"
		#define NetworkLog(status,method,mns) pankey_Log(status,"Network",method,mns)
	#else
		#define NetworkLog(status,method,mns)
	#endif

	namespace pankey{

		namespace AI{

			struct Network{

				Network() = delete;
				Network(int a_target_size, int a_source_size, int a_wight_size){
					m_target.createFastArray(a_target_size);
					m_source.createFastArray(a_source_size);
					m_weightIndex.createFastArray(a_wight_size);
				}

				Base::ArrayPointer<uint32_t,uint32_t> m_source;
				Base::ArrayPointer<uint32_t,uint32_t> m_target;

				Base::ArrayPointer<uint32_t,uint32_t> m_weightIndex;

				uint32_t length() const{
					return m_source.length();
				}
				
			};

		}

	}

#endif
