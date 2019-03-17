PROJECT(FashionMnist)

SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -pg -O2")

SET(SRC FashionMnist/FashionMnistData.cpp
        FashionMnist/FashionMnist.cpp)

ADD_EXECUTABLE(FashionMnist ${SRC})
TARGET_LINK_LIBRARIES(FashionMnist "${TORCH_LIBRARIES}")
