cmake_minimum_required(VERSION 3.13)
project(EXAMTD C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ftree-vectorize -flto")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")

add_executable(EXAMTD main.c cgradient.h cgradient.c matrix.h kinds.h matrix.c precond.h precond.c)
target_link_libraries(EXAMTD lapacke cblas)
