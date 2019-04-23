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
#include "ReadFile.hpp"


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

      void CIFAR100::SaveTensors(const std::vector<int> &lables, const std::vector<uint8_t> &imgs)
      {
        int size = lables.size();
        auto imgTensor = torch::tensor(imgs);
        Images = imgTensor.to(torch::kFloat).div_(255);
        Images = Images.view({size, CHANNELS, ROWS, COLUMNS});

        auto targTensor = torch::tensor(lables);
        Targets = targTensor.view({size});

        return;
      }

      void CIFAR100::ReadLableFile(const std::string &path)
      {
        std::string LABELS_FILE("batches.meta.txt");
        std::string file(path + LABELS_FILE);
        std::cout << "\nProcessing Lables file : " << file.c_str() << "\n";

        ReadFile rFile(file);
        LabelsVector = rFile.Data();

        for (auto &v: LabelsVector) {
          std::cout << v << "\n";
        }
        std::cout << "\n";
      }

      void CIFAR100::ReadBinFile(const std::string &path, bool mode)
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

      std::tuple<std::vector<int>, std::vector<uint8_t>> CIFAR100::GetData(const std::string &fileName)
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

      CIFAR100::CIFAR100(const std::string &root, Mode mode)
      {
        ReadBinFile(root, Mode::kTrain == mode);
        ReadLableFile(root);
      }

      CIFAR100::~CIFAR100()
      {
        // Empty.
      }

      Example<> CIFAR100::get(size_t index) {
        return {Images[index], Targets[index]};
      }

      optional<size_t> CIFAR100::size() const
      {
        return Images.size(0);
      }

      std::string CIFAR100::GetTarget(int id)
      {
        if (LabelsVector.empty()) {
          std::cout << "Something is wrong with the lables file..\n";
          return std::string("Unknown");
        }
        if (id > static_cast<int>(LabelsVector.size()) || id < 0) {
          return std::string("Unknown");
        }
        return LabelsVector[id];
      }

      bool CIFAR100::IsTrain() const noexcept
      {
        return Images.size(0) == NUM_TRAIN_FILES * DATA_SIZE;
      }

      const Tensor& CIFAR100::GetImages() const
      {
        return Images;
      }

      const Tensor& CIFAR100::GetTargets() const
      {
        return Targets;
      }

    } // ns datasets
  } // ns data
} // ns torch

