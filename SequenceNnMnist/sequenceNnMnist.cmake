PROJECT(SequenceNnMnist)

ADD_EXECUTABLE(SequenceNnMnist SequenceNnMnist/SequenceNnMnist.cpp)
TARGET_LINK_LIBRARIES(SequenceNnMnist "${TORCH_LIBRARIES}")
