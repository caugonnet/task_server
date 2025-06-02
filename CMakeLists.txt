cmake_minimum_required(VERSION 3.18)
project(task_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(server server.cu)
target_link_libraries(server PRIVATE cuda cudart)

add_executable(client client.cu)
