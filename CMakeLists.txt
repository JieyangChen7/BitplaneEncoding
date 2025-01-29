cmake_minimum_required(VERSION 3.19)

project(
  BitplaneEncoding
  LANGUAGES CUDA
)

set(CMAKE_CUDA_ARCHITECTURES "70")

add_executable(test test.cu)