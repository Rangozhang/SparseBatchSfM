cmake_minimum_required( VERSION 2.8 )
project( SparseBatchSfM )
# set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package( OpenCV REQUIRED )
find_package( Ceres REQUIRED )
# find_package( Glog REQUIRED )
# find_package( Threads REQUIRED )

# MESSAGE(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

set(src "src")
set(inc "include")
include_directories(${inc})
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/eigen)

add_library(Util ${inc}/utils.hpp ${src}/utils.cpp)

add_library(ImageCapture ${inc}/ImageCapture.hpp ${src}/ImageCapture.cpp)

add_library(Protos ${inc}/protos.hpp)
set_target_properties(Protos PROPERTIES LINKER_LANGUAGE CXX)

add_library(FeatureProcessor ${inc}/FeatureProcessor.hpp ${src}/FeatureProcessor.cpp)
target_link_libraries(FeatureProcessor ${OpenCV_LIBS})

add_library(BundleAdjustment ${inc}/BundleAdjustment.hpp ${src}/BundleAdjustment.cpp)
target_link_libraries(BundleAdjustment ceres ${GLOG_LIBRARIES})

add_library(TwoViewReconstruction ${inc}/TwoViewReconstruction.hpp ${src}/TwoViewReconstruction.cpp)
target_link_libraries(TwoViewReconstruction ${OpenCV_LIBS})
#target_link_libraries(TwoViewReconstruction BundleAdjustment)

add_library(GraphMerge ${inc}/GraphMerge.hpp ${src}/GraphMerge.cpp)

add_library(SparseBatchSfM ${inc}/SparseBatchSfM.hpp ${src}/SparseBatchSfM.cpp)

# add_executable(main ${src}/main.cpp ${inc}/utils.hpp ${src}/utils.cpp ${inc}/ImageCapture.hpp ${src}/ImageCapture.cpp ${inc}/protos.hpp ${inc}/FeatureProcessor.hpp ${src}/FeatureProcessor.cpp ${inc}/BundleAdjustment.hpp ${src}/BundleAdjustment.cpp ${inc}/TwoViewReconstruction.hpp ${src}/TwoViewReconstruction.cpp ${inc}/SparseBatchSfM.hpp ${src}/SparseBatchSfM.cpp)
add_executable(main ${src}/main.cpp)
# target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} glog SparseBatchSfM ImageCapture Util Protos FeatureProcessor TwoViewReconstruction BundleAdjustment)
target_link_libraries(main ${OpenCV_LIBS} ceres ${GLOG_LIBRARIES} SparseBatchSfM ImageCapture Util Protos FeatureProcessor TwoViewReconstruction BundleAdjustment GraphMerge)
# target_link_libraries(main ${OpenCV_LIBS} ceres ${CERES_LIBRARIES} ${GLOG_LIBRARIES})
add_definitions(-std=c++11)
