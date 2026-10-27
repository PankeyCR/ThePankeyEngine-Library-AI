
#ifndef AGIModel_hpp
	#define AGIModel_hpp

	#include "AGI.hpp"

	#if defined(pankey_Log) && (defined(AGIModel_Log) || defined(pankey_Global_Log) || defined(pankey_ArduinoIO_Log))
		#include "Logger_status.hpp"
		#define AGIModelLog(status,method,mns) pankey_Log(status,"AGIModel",method,mns)
	#else
		#define AGIModelLog(status,method,mns)
	#endif

	namespace pankey{

		namespace AI{

			struct AGIModel{
				using STATE_TYPE = AGI;
				
				using INITIALIZEMODEL_TYPE = AGIModel;
				using UPDATEMODEL_TYPE = AGIModel;

				using WRITEMODEL_TYPE = AGIModel;
				using EXPORTER_TYPE = AGIModel;

				using READMODEL_TYPE = AGIModel;
				using IMPORTER_TYPE = AGIModel;
			};

			template<class S, class A>
			void initializeState(AGIModel a_model, S& a_state, A& a_app){
				AGIModelLog(pankey_Log_StartMethod, "initializeState", "");
				AGIModelLog(pankey_Log_EndMethod, "initializeState", "");
			}

			template<class S, class A, class... Args>
			void updateState(AGIModel a_model, S& a_state, A& a_app, Args... a_args){
				AGIModelLog(pankey_Log_StartMethod, "updateState", "");
				AGIModelLog(pankey_Log_EndMethod, "updateState", "");
			}

			template<class S, class A, class E>
			void writeState(AGIModel a_model, S& a_state, A& a_ip, E& a_exporter){

			}

			template<class S, class A, class I>
			void readState(AGIModel a_model, S& a_state, A& a_ip, I& a_importer){

			}

		}

	}

#endif