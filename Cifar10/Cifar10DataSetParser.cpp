#include <torch/data/datasets/mnist.h>

#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

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

      void CIFAR10::SplitDump(const std::string &file)
      {
        std::ifstream fileBuf(file, std::ios::binary);
        AT_CHECK(fileBuf, "Error opening images file at ", file);
        //fileBuf.seekg(0, fileBuf.end);
        //std::cout << "Len = " << fileBuf.tellg() << "\n";
#if 1
        int count(0);
        while (fileBuf) {
          count++;
          //std::ios::pos_type before = fileBuf.tellg();
          //uint8_t x;
          uint8_t x;
          fileBuf >> x;
          //std::ios::pos_type after = fileBuf.tellg();
          /*std::cout << before << ' ' << static_cast<int>(x) << ' '
            << after << std::endl;*/
          std::cout << static_cast<int>(x) << " ";
          if (count % 1024 == 0) {
            std::cout << "\n\n\n";
          }
          if (count > 3073) {
            std::cout << "\n\n\n";
            break;
          }
        }
        std::cout << "Count = " << count << "\n";
#endif
        return;
      }


      // Public methods

      CIFAR10::CIFAR10(const std::string &root, Mode mode)
      {
        ReadBinFile(root, mode == Mode::kTrain);
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

