cmake_minimum_required(VERSION 3.22)
project(LinearRegressionQuantizationCpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(LinearRegressionQuantizationCpp main.cpp QLinearRegression.cpp QLinearRegression.h quantization.h quantization.cpp IntLinearRegression.cpp IntLinearRegression.h)
