PROJECT(Cifar10Classifier)

#SET(OpenCV_DIR /usr/lib/cmake/opencv4)
SET(OpenCV_DIR /usr/lib/cmake/openCV-3.4.3)
#SET(OpenCV_DIR /home/santhosh/course/reloaded/opencv/new_custom_install/lib/cmake/opencv4)
FIND_PACKAGE(OpenCV REQUIRED)


SET(SRC Cifar10/Cifar10DataSetParser.cpp
        Cifar10/Cifar10.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(Cifar10 ${SRC})
TARGET_LINK_LIBRARIES(Cifar10  ${OpenCV_LIBS} ${TORCH_LIBRARIES} )
