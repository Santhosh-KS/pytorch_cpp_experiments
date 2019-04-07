PROJECT(Cifar10SaveAndLoad)

SET(SRC Cifar10SaveAndLoad/main.cpp
        Common/Cifar10DataSetParser.cpp)

SET(BIN Cifar10SaveAndLoad)
SET(INC Common)

INCLUDE_DIRECTORIES(${INC})

ADD_EXECUTABLE(${BIN} ${SRC})
TARGET_LINK_LIBRARIES(${BIN} ${TORCH_LIBRARIES})
