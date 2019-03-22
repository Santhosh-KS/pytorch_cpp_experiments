PROJECT(Cifar10Classifier)

SET(SRC Cifar10/Cifar10DataSetParser.cpp
        Cifar10/Cifar10.cpp)
      ADD_EXECUTABLE(Cifar10 ${SRC})
TARGET_LINK_LIBRARIES(Cifar10 "${TORCH_LIBRARIES}")
