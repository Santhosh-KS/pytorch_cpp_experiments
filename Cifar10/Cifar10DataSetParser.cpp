#include <torch/data/datasets/mnist.h>

#include <torch/data/example.h>
#include <torch/types.h>

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
          for (int i = 1; i < numFiles; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            std::cout << file.c_str() << "\n";
            FastSeek(file);
          }
        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          FastSeek(file);
        }
      }

      void CIFAR10::FastSeek(const std::string &fileName)
      {
        constexpr uint32_t dataSize(10000);
        constexpr uint32_t numElements(3*32*32);
        std::ifstream file;

        try {
          file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);

          AT_CHECK(file, "Error opening images file at ", fileName);

          std::cout << "Processing file : " << fileName.c_str() << "\n\n";

          torch::Tensor testLabels = torch::empty({dataSize,1},  torch::kByte);

          size_t pos = 0;
          for (uint32_t i =0; i < dataSize; i++) {
            file.seekg(pos);
            char buffer;
            file.read(&buffer, 1);
            testLabels[i]= static_cast<int>(buffer);
            pos += 1;
            file.seekg(pos);
            torch::Tensor testImg = torch::empty({1, numElements});
            file.read(reinterpret_cast<char*>(testImg.data_ptr()), numElements);
            pos += numElements;
            testImg.to(torch::kFloat32).div_(255);
          }

          for (int i = 0; i < 10 ; i++) {
            std::cout << testLabels[i][0] << "\n";
          }
        }
        catch(...) {
          file.close();
        }
        file.close();
        return ;
      }
#if 0 // Very slow
      void CIFAR10::SplitDump(const std::string &filePath)
      {
        std::ifstream file;
        file.open(filePath, std::ios::in | std::ios::binary | std::ios::ate);

        AT_CHECK(file, "Error opening images file at ", filePath);


        auto fileSize = file.tellg();
        auto buffer = std::make_unique<char[]>(fileSize);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), fileSize);
        file.close();

        std::cout << "File Size = " << fileSize << "\n";
        //std::vector<std::vector<int>> images;
        //std::vector<int> labels;
        std::size_t size(10000);
        //images.reserve(size);
        // labels.resize(size);
        torch::Tensor test = torch::empty({10000,1});

        for(std::size_t i = 0; i < size; ++i) {
          //  labels[i] = static_cast<int>(buffer[i * 3073]);
          test[i] = static_cast<int>(buffer[i * 3073]);

          //std::vector<int> image;
          // image.resize(3*32*32);
          //images.push_back(image);
          torch::Tensor testImg = torch::empty({10000,3*32*32});

          for(std::size_t j = 1; j < 3073; ++j) {
            // images[i][j - 1] = static_cast<int>(buffer[i * 3073 + j]);
            testImg[i][j - 1] = static_cast<int>(buffer[i * 3073 + j]);
          }
          if (i < 5) {
            std::cout << "tensor dim = " << testImg.numel() << "\n";
          }
        }
        //std::cout << "Images size = " << images.size() << "\n";
        //std::cout << "Lables size = " << labels.size() << "\n";
        //for(int i =0 ; i < 10 ; i++) {
        //  std::cout << labels[i] << "\n";
        // }
        // torch::Tensor lableTensor = torch::from_blob(labels, {size,1});
        return;
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
        std::cout << index << "\n";
        torch::data::Example<> data;
        return data;
      }

      optional<size_t> CIFAR10::size() const
      {
        return Images_.size(0);
      }
    } // ns datasets
  } // ns data
} // ns torch

