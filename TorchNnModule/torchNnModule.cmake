PROJECT(TorchNnModule)

ADD_EXECUTABLE(TorchNnModule TorchNnModule/TorchNnModule.cpp)
TARGET_LINK_LIBRARIES(TorchNnModule "${TORCH_LIBRARIES}")
