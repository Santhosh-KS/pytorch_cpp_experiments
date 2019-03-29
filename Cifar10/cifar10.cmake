PROJECT(Cifar10Classifier)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

SET(SRC Cifar10/Cifar10DataSetParser.cpp
        Cifar10/Cifar10.cpp)

ADD_EXECUTABLE(Cifar10 ${SRC})
TARGET_LINK_LIBRARIES(Cifar10 ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} 
