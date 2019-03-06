PROJECT(HelloAten-App)

ADD_EXECUTABLE(HelloAten HelloWorld/HelloAten.cpp)
TARGET_LINK_LIBRARIES(HelloAten "${TORCH_LIBRARIES}")
