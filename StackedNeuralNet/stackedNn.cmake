PROJECT(StackedNeuralNet)

ADD_EXECUTABLE(StackedNeuralNet StackedNeuralNet/StackedNeuralNet.cpp)
TARGET_LINK_LIBRARIES(StackedNeuralNet "${TORCH_LIBRARIES}")
