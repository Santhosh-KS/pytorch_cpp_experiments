CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

# if we want to profile all the bins just enable the below line with -pg option.
#SET(PROFILE_FLAGS "-pg -O2")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_COMPILE_CUSTOM_FLAGS} ${PROFILE_FLAGS}")

SET(OPENCV_INCLUDED NO)
SET(CPP_COMPILE_CUSTOM_FLAGS "-Wall -Wextra -Wpedantic -Werror")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_COMPILE_CUSTOM_FLAGS}")
SET(CMAKE_PREFIX_PATH /opt/libtorch)

FIND_PACKAGE(Torch REQUIRED)

INCLUDE(HelloWorld/HelloWorld.cmake)
INCLUDE(FrontEndExample/FrontEnd.cmake)
INCLUDE(SimpleNeuralNet/simpleNn.cmake)
INCLUDE(StackedNeuralNet/stackedNn.cmake)
INCLUDE(BasicMnist/basicMnist.cmake)
INCLUDE(TorchNnModule/torchNnModule.cmake)
INCLUDE(SequenceNnMnist/sequenceNnMnist.cmake)
INCLUDE(NnMnist/nnMnist.cmake)
INCLUDE(SaveAndLoadNnMnist/saveAndLoadNnMnist.cmake)
INCLUDE(FashionMnist/fashionMnist.cmake)
INCLUDE(CatsAndDogsCnn/catsAndDogsCnn.cmake)
INCLUDE(Cifar10/cifar10.cmake)
INCLUDE(Cifar10SaveAndLoad/cifar10SaveAndLoad.cmake)
#INCLUDE(testModel/testModel.cmake)
