PROJECT(Cifar10Classifier)

#SET(IMG_DISPLAY_ENABLED 1)

SET(SRC Cifar10/Cifar10DataSetParser.cpp
        Cifar10/Cifar10.cpp)

#if (IMG_DISPLAY_ENABLED)
#SET(OpenCV_DIR /usr/lib)
#  SET(OpenCV_DIR /home/santhosh/course/reloaded/opencv/new_custom_install/lib/cmake/opencv4)
#FIND_PACKAGE(OpenCV REQUIRED)
#INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
#endif (IMG_DISPLAY_ENABLED)

ADD_EXECUTABLE(Cifar10 ${SRC})
TARGET_LINK_LIBRARIES(Cifar10  ${OpenCV_LIBS} ${TORCH_LIBRARIES})
