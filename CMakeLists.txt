cmake_minimum_required(VERSION 3.13)
project(EXAMTD C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ftree-vectorize -flto")

add_executable(EXAMTD src/main.c src/cgradient.h src/cgradient.c src/matrix.h src/kinds.h src/matrix.c src/precond.h src/precond.c)
target_link_libraries(EXAMTD lapacke cblas m)
