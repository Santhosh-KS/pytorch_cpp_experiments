PROJECT(FrontEnd-App)

#SET(CMAKE_CXX_STANDARD 11)
#SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#SET(CMAKE_PREFIX_PATH /opt/libtorch)

#FIND_PACKAGE(Torch REQUIRED)
ADD_EXECUTABLE(FrontEnd FrontEndExample/FrontEndExample.cpp)
TARGET_LINK_LIBRARIES(FrontEnd "${TORCH_LIBRARIES}")