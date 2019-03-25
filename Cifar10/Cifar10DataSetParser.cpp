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
          }
        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          SplitDump(file);
        }
      }

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
        std::vector<std::vector<int>> images;
        std::vector<int> labels;
        std::size_t size(10000);
        images.reserve(size);
        labels.resize(size);

        for(std::size_t i = 0; i < size; ++i) {
          labels[i] = static_cast<int>(buffer[i * 3073]);

          std::vector<int> image;
          image.resize(3*32*32);
          images.push_back(image);

          for(std::size_t j = 1; j < 3073; ++j) {
            images[i][j - 1] = static_cast<int>(buffer[i * 3073 + j]);
          }
        }
        std::cout << "Images size = " << images.size() << "\n";
        std::cout << "Lables size = " << labels.size() << "\n";
        for(int i =0 ; i < 10 ; i++) {
          std::cout << labels[i] << "\n";
        }
        return;
      }


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

