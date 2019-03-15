PROJECT(SaveAndLoadNnMnist)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -pg -O2")

ADD_EXECUTABLE(SaveAndLoadNnMnist SaveAndLoadNnMnist/SaveAndLoadNnMnist.cpp)
TARGET_LINK_LIBRARIES(SaveAndLoadNnMnist "${TORCH_LIBRARIES}")
