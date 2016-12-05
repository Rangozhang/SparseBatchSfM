cmake_minimum_required( VERSION 2.8 )
project( SparseBatchSfM )
find_package( OpenCV REQUIRED )

set(src "src")
set(inc "include")
include_directories(${inc})
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/eigen)

add_library(Util ${inc}/utils.hpp ${src}/utils.cpp)

add_library(ImageCapture ${inc}/ImageCapture.hpp ${src}/ImageCapture.cpp)

add_library(SparseBatchSfM ${inc}/SparseBatchSfM.hpp ${src}/SparseBatchSfM.cpp)

add_library(Protos ${inc}/protos.hpp)
set_target_properties(Protos PROPERTIES LINKER_LANGUAGE CXX)

add_library(FEATUREPROCESSOR ${inc}/FeatureProcessor.hpp ${src}/FeatureProcessor.cpp)

add_library(TWOVIEWRECONSTRUCTION ${inc}/TwoViewReconstruction.hpp ${src}/TwoViewReconstruction.cpp)

add_executable(main ${src}/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} SparseBatchSfM ImageCapture Util Protos)
add_definitions(-std=c++11)
