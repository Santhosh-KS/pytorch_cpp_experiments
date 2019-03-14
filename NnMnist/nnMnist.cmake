PROJECT(NnMnist)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -pg -O2")

ADD_EXECUTABLE(NnMnist NnMnist/NnMnist.cpp)
TARGET_LINK_LIBRARIES(NnMnist "${TORCH_LIBRARIES}")
