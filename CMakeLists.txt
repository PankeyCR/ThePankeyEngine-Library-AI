cmake_minimum_required(VERSION 3.15)

project(ThePankeyEngine_Library_AI)

if(NOT TARGET ThePankeyEngine_Library_AI)

    add_library(ThePankeyEngine_Library_AI INTERFACE)

    target_include_directories(
        ThePankeyEngine_Library_AI
          INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

endif()