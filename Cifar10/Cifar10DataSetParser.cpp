#include <torch/data/datasets/mnist.h>

#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/torch.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <memory>
#include <algorithm>

#include "Cifar10DataSetParser.hpp"


namespace torch {
  namespace data {
    namespace datasets {

      // Private Methods.
      void CIFAR10::ReadBinFile(const std::string &path, bool mode)
      {
        std::string suffix(".bin");
        std::string prefix("data_batch_");
        std::string file("");

        if (mode) { // train data.
          std::vector<torch::Tensor> labelsVector;
          //std::vector<char> imagesVector;
          std::vector<torch::Tensor> imagesVector;

          /* The data is stored as bin in 4 different file for training set and in 1 file for test set.
           * Data is stored as follows for both the datasets.
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * ....
           * */
          uint8_t numFiles = 5;
          for (int i = 1; i < numFiles; i++) {
          //for (int i = 1; i < 2; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            torch::Tensor tmpLabel;
            torch::Tensor tmpImg;
            std::tie(tmpLabel, tmpImg) = GetData(file);
            labelsVector.push_back(tmpLabel);
            imagesVector.push_back(tmpImg);

#if 0
            auto minMax = std::minmax_element (tmpImg.begin(), tmpImg.end());
            std::cout << "Tmp vec = " << tmpImg.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#endif
          }


          Targets = labelsVector[0];
          Images = imagesVector[0];
          for (int i = 1; i < numFiles; i++) {
            Images = torch::cat({Images, imagesVector[i]},0);
            Targets = torch::cat({Targets, labelsVector[i]},0);
            std::cout << "Concating Image tensors .. num ele = " << Images.numel() << "\n";
            std::cout << "Concating Target tensors .. num ele = " << Targets.numel() << "\n";
          }

#if 0
          for (int i = 1; i < 2; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            //std::cout << file.c_str() << "\n";
            std::vector<int> tmpLabel;
            //std::vector<char> tmpImg;
            std::vector<uint8_t> tmpImg;
            std::tie(tmpLabel, tmpImg) = GetData(file);
            labelsVector.resize(labelsVector.size()+tmpLabel.size());
            labelsVector.insert(labelsVector.end(), tmpLabel.begin(), tmpLabel.end());
            imagesVector.resize(imagesVector.size()+tmpImg.size());
            imagesVector.insert(imagesVector.end(), tmpImg.begin(), tmpImg.end());

#if 0
            auto minMax = std::minmax_element (tmpImg.begin(), tmpImg.end());
            std::cout << "Tmp vec = " << tmpImg.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#endif
          }
#endif
          //std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";
          int size = labelsVector.size();
#if 0
          torch::Tensor tmpTensor = torch::empty({size, 3, 32, 32}, torch::kByte);
          Images = tmpTensor.view({size*3*32*32});

          for (size_t i = 0; i < imagesVector.size(); i++) {
            Images[i] = imagesVector[i];
          }
          Images = Images.view({size,3*32*32}).to(kFloat32).div_(255);


#endif
          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          //std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
          //std::cout << "Target size = " << Targets.sizes() << "\n";
        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          std::tie(Targets, Images) = GetData(file);
          std::cout << "Image tensors .. num ele = " << Images.numel() << "\n";
          std::cout << "Target tensors .. num ele = " << Targets.numel() << "\n";
#if 0
          std::vector<int> labelsVector;
          //std::vector<char> imagesVector;
          std::vector<uint8_t> imagesVector;
          std::tie(labelsVector, imagesVector) = GetData(file);
          //std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";

          int size = labelsVector.size();

          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          //std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
#endif
          //std::cout << "Target size = " << Targets.sizes() << "\n";
        }
      }

      //std::tuple<std::vector<int>, std::vector<char>> CIFAR10::GetData(const std::string &fileName)
      //std::tuple<std::vector<int>, std::vector<uint8_t>> CIFAR10::GetData(const std::string &fileName)
      std::tuple<torch::Tensor, torch::Tensor> CIFAR10::GetData(const std::string &fileName)
      {
        constexpr uint32_t dataSize(10000);
        constexpr uint32_t channels(3);
        constexpr uint32_t rows(32);
        constexpr uint32_t columns(32);
        constexpr uint32_t numElements(channels*rows*columns);


        std::vector<int> labelVector;
        labelVector.resize(dataSize);
        torch::Tensor labelTensor = torch::empty({dataSize},torch::kByte);
        //std::vector<char> imgVector;
        std::vector<uint8_t> imgVector;
        imgVector.reserve(dataSize*numElements);
        std::vector<torch::Tensor> imgTensor;
        imgTensor.reserve(dataSize);

        torch::Tensor concatTensor;

        std::ifstream file;

        try {
          file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);

          AT_CHECK(file, "Error opening images file at ", fileName);

          size_t pos = 0;
          for (uint32_t i =0; i < dataSize; i++) {
            file.seekg(pos);
            char buffer;
            file.read(&buffer, 1);
            labelVector[i]= static_cast<int>(buffer);
            labelTensor[i]= static_cast<int>(buffer);
            pos += 1;
            file.seekg(pos);
            //auto imgBuf = std::make_unique<char[]>(numElements);
            char imgBuf[numElements]={0};

            //file.read(imgBuf.get(), numElements);
            file.read(imgBuf, numElements);
            auto tmpTensor = torch::empty({numElements},torch::kByte);
            file.read(reinterpret_cast<char*>(tmpTensor.data_ptr()), tmpTensor.numel());
            imgTensor.push_back(tmpTensor);
            pos += numElements;

#if 0
            std::vector<uint8_t> testVec;
            testVec.reserve(numElements);
            testVec.insert(testVec.end(), &imgBuf[0], &imgBuf[0]+numElements);
            auto minMax = std::minmax_element (testVec.begin(), testVec.end());
            std::cout << "Test Vec = " << testVec.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#endif
            //imgVector.insert(imgVector.end(), imgBuf.get(), imgBuf.get()+numElements);
            imgVector.insert(imgVector.end(), &imgBuf[0], &imgBuf[0]+numElements);
          }
#if 1
          concatTensor = imgTensor[0];
          for(size_t i = 1; i < imgTensor.size(); i++) {
            concatTensor = torch::cat({concatTensor, imgTensor[i]}, 0);
            std::cout << "Concating tensor.. num ele = " << concatTensor.numel() << "\n";
          }
#endif
          file.close();
          std::cout << "Done Processing file : " << fileName.c_str() << "\n";
        }
        catch(...) {
          std::cout << "Exception occoured!\n";
          file.close();
        }
        //return std::make_tuple(labelVector, imgVector);
        return std::make_tuple(labelTensor, concatTensor);
      }
#if 0

      void CIFAR10::ReadBinFile(const std::string &path, bool mode)
      {
        std::string suffix(".bin");
        std::string prefix("data_batch_");
        std::string file("");

        if (mode) { // train data.
          std::vector<int> labelsVector;
          //std::vector<char> imagesVector;
          std::vector<uint8_t> imagesVector;

          /* The data is stored as bin in 4 different file for training set and in 1 file for test set.
           * Data is stored as follows for both the datasets.
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * ....
           * */
          //uint8_t numFiles = 5;
          //for (int i = 1; i < numFiles; i++) {
          for (int i = 1; i < 2; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            //std::cout << file.c_str() << "\n";
            std::vector<int> tmpLabel;
            //std::vector<char> tmpImg;
            std::vector<uint8_t> tmpImg;
            std::tie(tmpLabel, tmpImg) = GetData(file);
            labelsVector.resize(labelsVector.size()+tmpLabel.size());
            labelsVector.insert(labelsVector.end(), tmpLabel.begin(), tmpLabel.end());
            imagesVector.resize(imagesVector.size()+tmpImg.size());
            imagesVector.insert(imagesVector.end(), tmpImg.begin(), tmpImg.end());

#if 0
            auto minMax = std::minmax_element (tmpImg.begin(), tmpImg.end());
            std::cout << "Tmp vec = " << tmpImg.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#endif
          }
          //std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";
          int size = labelsVector.size();
          torch::Tensor tmpTensor = torch::empty({size, 3, 32, 32}, torch::kByte);
          Images = tmpTensor.view({size*3*32*32});

          for (size_t i = 0; i < imagesVector.size(); i++) {
            Images[i] = imagesVector[i];
          }
          Images = Images.view({size,3*32*32}).to(kFloat32).div_(255);

#if 0

          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          //std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
          //std::cout << "Target size = " << Targets.sizes() << "\n";
#endif
        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          std::vector<int> labelsVector;
          //std::vector<char> imagesVector;
          std::vector<uint8_t> imagesVector;
          std::tie(labelsVector, imagesVector) = GetData(file);
          //std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";

          int size = labelsVector.size();

          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          //std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
          //std::cout << "Target size = " << Targets.sizes() << "\n";
        }
      }

      //std::tuple<std::vector<int>, std::vector<char>> CIFAR10::GetData(const std::string &fileName)
      std::tuple<std::vector<int>, std::vector<uint8_t>> CIFAR10::GetData(const std::string &fileName)
      {
        constexpr uint32_t dataSize(10000);
        constexpr uint32_t channels(3);
        constexpr uint32_t rows(32);
        constexpr uint32_t columns(32);
        constexpr uint32_t numElements(channels*rows*columns);


        std::vector<int> labelVector;
        labelVector.resize(dataSize);
        //std::vector<char> imgVector;
        std::vector<uint8_t> imgVector;
        imgVector.reserve(dataSize*numElements);

        std::ifstream file;

        try {
          file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);

          AT_CHECK(file, "Error opening images file at ", fileName);

          size_t pos = 0;
          for (uint32_t i =0; i < dataSize; i++) {
            file.seekg(pos);
            char buffer;
            file.read(&buffer, 1);
            labelVector[i]= static_cast<int>(buffer);
            pos += 1;
            file.seekg(pos);
            //auto imgBuf = std::make_unique<char[]>(numElements);
            char imgBuf[numElements]={0};

            //file.read(imgBuf.get(), numElements);
            file.read(imgBuf, numElements);
            pos += numElements;

#if 0
            std::vector<uint8_t> testVec;
            testVec.reserve(numElements);
            testVec.insert(testVec.end(), &imgBuf[0], &imgBuf[0]+numElements);
            auto minMax = std::minmax_element (testVec.begin(), testVec.end());
            std::cout << "Test Vec = " << testVec.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#endif
            //imgVector.insert(imgVector.end(), imgBuf.get(), imgBuf.get()+numElements);
            imgVector.insert(imgVector.end(), &imgBuf[0], &imgBuf[0]+numElements);
          }
          file.close();
          std::cout << "Done Processing file : " << fileName.c_str() << "\n";
        }
        catch(...) {
          std::cout << "Exception occoured!\n";
          file.close();
        }
        return std::make_tuple(labelVector, imgVector);
      }
#endif
      // Public methods

      CIFAR10::CIFAR10(const std::string &root, Mode mode)
      {
        ReadBinFile(root, Mode::kTrain == mode);
      }

      CIFAR10::~CIFAR10()
      {
        // Empty.
      }

      Example<> CIFAR10::get(size_t index) {
        return {Images[index], Targets[index]};
      }

      optional<size_t> CIFAR10::size() const
      {
        return Images.size(0);
      }
    } // ns datasets
  } // ns data
} // ns torch

