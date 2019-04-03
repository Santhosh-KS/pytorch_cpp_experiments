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

      // Number of images per file.
      constexpr uint32_t DATA_SIZE(10000);
      constexpr uint32_t CHANNELS(3);
      constexpr uint32_t ROWS(32);
      constexpr uint32_t COLUMNS(32);
      constexpr uint32_t NUM_ELEMENTS(CHANNELS * ROWS * COLUMNS); // 3072
      constexpr uint8_t NUM_TRAIN_FILES(5);

      // Private Methods.

      void CIFAR10::SaveTensors(const std::vector<int> &lables, const std::vector<uint8_t> &imgs)
      {
        int size = lables.size();
        auto imgTensor = torch::tensor(imgs);
        Images = imgTensor.to(torch::kFloat).div_(255);
        Images = Images.view({size, CHANNELS, ROWS, COLUMNS});

        auto targTensor = torch::tensor(lables);
        Targets = targTensor.view({size});

        return;
      }

      void CIFAR10::ReadBinFile(const std::string &path, bool mode)
      {
        std::string suffix(".bin");
        std::string prefix("data_batch_");
        std::string file("");

        std::vector<int> labelsVector;
        std::vector<uint8_t> imagesVector;
        if (mode) { // train data.

          /* The data is stored as bin in 4 different file for training set and in 1 file for test set.
           * Data is stored as follows for both the datasets.
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * <lable - 1 byte><RGB channel - 3072 bytes>
           * ....
           * */

          for (int i = 1; i <= NUM_TRAIN_FILES; i++) {
            file = path + prefix + std::to_string(i) + suffix;
            std::vector<int> tmpLabel;
            std::vector<uint8_t> tmpImg;
            std::tie(tmpLabel, tmpImg) = GetData(file);
            labelsVector.insert(labelsVector.end(), tmpLabel.begin(), tmpLabel.end());
            imagesVector.insert(imagesVector.end(), tmpImg.begin(), tmpImg.end());
          }
          SaveTensors(labelsVector, imagesVector);
        }
        else { // Test data
          prefix.clear();
          prefix = "test_batch";
          file = path + prefix + suffix;
          std::cout << file.c_str() << "\n";
          std::tie(labelsVector, imagesVector) = GetData(file);
          SaveTensors(labelsVector, imagesVector);
        }
      }

      std::tuple<std::vector<int>, std::vector<uint8_t>> CIFAR10::GetData(const std::string &fileName)
      {
        std::vector<int> labelVector;
        labelVector.resize(DATA_SIZE);
        std::vector<uint8_t> imgVector;
        imgVector.reserve(DATA_SIZE * NUM_ELEMENTS);

        std::ifstream file;

        try {
          file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);

          AT_CHECK(file, "Error opening images file at ", fileName);

          size_t pos = 0;
          for (uint32_t i =0; i < DATA_SIZE; i++) {
            file.seekg(pos);
            char buffer;
            file.read(&buffer, 1);
            labelVector[i]= static_cast<int>(buffer);
            pos += 1;
            file.seekg(pos);
            char imgBuf[NUM_ELEMENTS]={0};

            file.read(imgBuf, NUM_ELEMENTS);
            pos += NUM_ELEMENTS;

            imgVector.insert(imgVector.end(), &imgBuf[0], &imgBuf[0] + NUM_ELEMENTS);
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

      std::string CIFAR10::GetTarget(int id)
      {
        std::string retVal("Unknown");
        switch(id) {
          case 0:
            retVal = "airplane";
            break;
          case 1:
            retVal = "automobile";
            break;
          case 2:
            retVal = "bird";
            break;
          case 3:
            retVal = "cat";
            break;
          case 4:
            retVal = "deer";
            break;
          case 5:
            retVal = "dog";
            break;
          case 6:
            retVal = "frog";
            break;
          case 7:
            retVal = "horse";
            break;
          case 8:
            retVal = "ship";
            break;
          case 9:
            retVal = "truck";
            break;
          default:
            break;
        }
        return retVal;
      }

      bool CIFAR10::IsTrain() const noexcept
      {
        return Images.size(0) == NUM_TRAIN_FILES * DATA_SIZE;
      }

      const Tensor& CIFAR10::GetImages() const
      {
        return Images;
      }

      const Tensor& CIFAR10::GetTargets() const
      {
        return Targets;
      }

    } // ns datasets
  } // ns data
} // ns torch

