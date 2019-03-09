PROJECT(SimpleNeuralNet)

ADD_EXECUTABLE(SimpleNeuralNet SimpleNeuralNet/SimpleNeuralNet.cpp)
TARGET_LINK_LIBRARIES(SimpleNeuralNet "${TORCH_LIBRARIES}")
