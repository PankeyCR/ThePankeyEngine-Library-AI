
#ifndef MinimalError_AS_hpp
	#define MinimalError_AS_hpp

	#include "BaseAppState.hpp"

	#if defined(pankey_Log) && (defined(MinimalError_AS_Log) || defined(pankey_Global_Log) || defined(pankey_Base_Log))
		#include "Logger_status.hpp"
		#define MinimalError_ASLog(status,method,mns) pankey_Log(status,"MinimalError_AS",method,mns)
	#else
		#define MinimalError_ASLog(status,method,mns)
	#endif

	namespace pankey{

		namespace Base{

			template<class A, class... Args>
			class MinimalError_AS : public BaseAppState<A,Args...>{
				public:
					MinimalError_AS(){}
					virtual ~MinimalError_AS(){}
					
					virtual void initialize(A& a_app){
						MinimalError_ASLog(pankey_Log_StartMethod, "initialize", "");
						m_init = true;
						initializeState(a_app);
						initializeState();
						MinimalError_ASLog(pankey_Log_EndMethod, "initialize", "");
					}
					
					virtual void initializeState(A& a_app){}
					
					virtual void initializeState(){}
					virtual bool hasInitialize(){return m_init;}
					
					virtual void onEnable(){m_enable = true;}
					virtual void onDisable(){m_enable = false;}
					
					virtual bool isEnable(){return m_enable;}
					
					
					virtual void update(A& a_app, Args... a_values){
						if(!m_enable){
							return;
						}
						updateState(a_app, a_values...);
					}
					
					virtual void updateState(A& a_app, Args... a_values){}

				protected:
					bool m_enable = false;
					bool m_init = false;
			};

		}

	}

#endif
