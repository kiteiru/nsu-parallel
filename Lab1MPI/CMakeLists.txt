cmake_minimum_required(VERSION 3.15)
project(Lab1MPI)

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{MSMPI_INC})
message(STATUS $ENV{MSMPI_INC})
find_package(MPI REQUIRED)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(Lab1MPI main.cpp)
add_executable(mpi1 mpi1.cpp)
add_executable(mpi2 mpi2.cpp)

set(LIB_ADDRESS "$ENV{MSMPI_LIB64}msmpi.lib")
string(REPLACE "\\" "/" LIBRARY ${LIB_ADDRESS}) # replace '\' with '/' for mingw
target_link_libraries(Lab1MPI ${LIBRARY})
target_link_libraries(mpi1 ${LIBRARY})
target_link_libraries(mpi2 ${LIBRARY})
