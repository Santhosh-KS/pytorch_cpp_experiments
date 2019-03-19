PROJECT(CatsAndDogsClassfier)

ADD_EXECUTABLE(CatsAndDogsCnn CatsAndDogsCnn/CatsAndDogsCnn.cpp)
TARGET_LINK_LIBRARIES(CatsAndDogsCnn "${TORCH_LIBRARIES}")
