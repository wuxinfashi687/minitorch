cmake_minimum_required(VERSION 3.28)
project(minitorch CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(Python_EXECUTABLE C:\\Users\\JUMPWORK-SERVER-01\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\minitorch-1q7o7QdB-py3.10\\Scripts\\python.exe)

find_package(Python3 COMPONENTS Interpreter Development)
link_libraries(${PYTHON_LIBRARIES})

include_directories(C:\\Users\\JUMPWORK-SERVER-01\\Desktop\\python_env\\py310\\include)
include_directories(C:\\Users\\JUMPWORK-SERVER-01\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\minitorch-1q7o7QdB-py3.10\\lib\\site-packages\\pybind11\\include)
include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\include)
link_directories(C:\\Users\\JUMPWORK-SERVER-01\\Desktop\\python_env\\py310\\libs)

add_executable(minitorch test_tensor.cu)

target_link_libraries(minitorch python310)