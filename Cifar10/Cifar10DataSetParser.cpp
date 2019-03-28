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
          uint8_t numFiles = 5;
          std::vector<int> labelsVector;
          std::vector<char> imagesVector;
          for (int i = 1; i < numFiles; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            std::cout << file.c_str() << "\n";
            std::vector<int> tmpLabel;
            std::vector<char> tmpImg;
            std::tie(tmpLabel, tmpImg) = GetData(file);
            labelsVector.resize(labelsVector.size()+tmpLabel.size());
            labelsVector.insert(labelsVector.end(), tmpLabel.begin(), tmpLabel.end());
            imagesVector.resize(imagesVector.size()+tmpImg.size());
            imagesVector.insert(imagesVector.end(), tmpImg.begin(), tmpImg.end());
          }
          std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";
          int size = labelsVector.size();

          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
          std::cout << "Target size = " << Targets.sizes() << "\n";

        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          std::vector<int> labelsVector;
          std::vector<char> imagesVector;
          std::tie(labelsVector, imagesVector) = GetData(file);
          std::cout << "labelsVector size = " << labelsVector.size() << " imagesVector.size() = " << imagesVector.size() <<"\n";

          int size = labelsVector.size();

          torch::Tensor tensorImage = torch::from_blob(&imagesVector[0], {size, 3, 32, 32}, torch::kByte);
          tensorImage = tensorImage.to(torch::kFloat).div_(255);
          Images = tensorImage;
          std::cout << "Images size = " << Images.sizes() << "\n";

          torch::Tensor tensorLabels = torch::from_blob(&labelsVector[0], {size}, torch::kInt32);
          Targets = tensorLabels;
          std::cout << "Target size = " << Targets.sizes() << "\n";
        }
      }

      std::tuple<std::vector<int>, std::vector<char>> CIFAR10::GetData(const std::string &fileName)
      {
        constexpr uint32_t dataSize(10000);
        constexpr uint32_t channels(3);
        constexpr uint32_t rows(32);
        constexpr uint32_t columns(32);
        constexpr uint32_t numElements(channels*rows*columns);


        std::vector<int> labelVector;
        labelVector.resize(dataSize);
        std::vector<char> imgVector;
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
            auto imgBuf = std::make_unique<char[]>(numElements);

            file.read(imgBuf.get(), numElements);
            pos += numElements;
            imgVector.insert(imgVector.end(), imgBuf.get(), imgBuf.get()+numElements);
          }
          file.close();
          std::cout << "\n Done Processing file : " << fileName.c_str() << "\n\n";
#if 0
          for (int i = 0; i < 10 ; i++) {
            std::cout << labelVector[i] << "\n";
          }
#endif
        }
        catch(...) {
          std::cout << "Exception occoured!\n";
          file.close();
        }
        return std::make_tuple(labelVector, imgVector);
      }
#if 0
      std::tuple<torch::Tensor, std::vector<torch::Tensor>> CIFAR10::FastSeek(const std::string &fileName)
      {
        constexpr uint32_t dataSize(10000);
        constexpr uint32_t channels(3);
        constexpr uint32_t rows(32);
        constexpr uint32_t columns(32);
        constexpr uint32_t numElements(channels*rows*columns);

        torch::Tensor lablesTensor = torch::empty({dataSize},  torch::kInt32);

        std::vector<torch::Tensor> imageTensorVector;
        imageTensorVector.reserve(dataSize);

        std::ifstream file;

        try {
          file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);

          AT_CHECK(file, "Error opening images file at ", fileName);

          size_t pos = 0;
          for (uint32_t i =0; i < dataSize; i++) {
            file.seekg(pos);
            char buffer;
            file.read(&buffer, 1);
            lablesTensor[i]= static_cast<int>(buffer);
            pos += 1;
            file.seekg(pos);
            //torch::Tensor imageTensor = torch::empty({1, channels, rows, columns});
            auto imageTensor = torch::empty({1, channels, rows, columns});
            file.read(reinterpret_cast<char*>(imageTensor.data_ptr()), numElements);
            pos += numElements;
            //imageTensorVector.push_back(std::move(imageTensor.to(torch::kFloat32).div_(255)));
            imageTensorVector.push_back(imageTensor.to(torch::kFloat32).div_(255));
          }
          std::cout << "\n Done Processing file : " << fileName.c_str() << "\n\n";
          file.close();
#if 0
          for (int i = 0; i < 10 ; i++) {
            std::cout << lablesTensor[i] << "\n";
          }
#endif
        }
        catch(...) {
          std::cout << "Exception occoured!\n";
          file.close();
        }
        return std::make_tuple(lablesTensor, imageTensorVector);
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

