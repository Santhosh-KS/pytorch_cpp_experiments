PROJECT(TestDeviceLoading)

SET(SRC testModel/main.cpp)

SET(BIN TestModel)

ADD_EXECUTABLE(${BIN} ${SRC})
TARGET_LINK_LIBRARIES(${BIN} ${TORCH_LIBRARIES})
