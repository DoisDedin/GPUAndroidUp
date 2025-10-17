cmake_minimum_required(VERSION 3.10.2)
project("vkfftlib")

set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=4096")

add_library(
        vkfftbridge
        SHARED
        vkfft_bridge.cpp
)

find_library(
        log-lib
        log
)

target_link_libraries(
        vkfftbridge
        ${log-lib}
)
