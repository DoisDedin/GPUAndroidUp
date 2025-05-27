cmake_minimum_required(VERSION 3.10.2)
project("vkfftlib")

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