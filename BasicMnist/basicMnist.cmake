PROJECT(BasicMnist)

ADD_EXECUTABLE(BasicMnist BasicMnist/BasicMnist.cpp)
TARGET_LINK_LIBRARIES(BasicMnist "${TORCH_LIBRARIES}")
