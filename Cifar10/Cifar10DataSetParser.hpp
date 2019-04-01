#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <cstddef>
#include <string>


namespace torch {
  namespace data {
    namespace datasets {

      namespace DS = torch::data::datasets;

      class CIFAR10 : public DS::Dataset<CIFAR10>
      {
        private:
          torch::Tensor Images;
          torch::Tensor Targets;

          void ReadBinFile(const std::string &path, bool mode);
          //std::tuple<torch::Tensor, std::vector<torch::Tensor>> FastSeek(const std::string &fileName);
          //std::tuple<std::vector<int>, std::vector<char>> GetData(const std::string &fileName);
          //std::tuple<std::vector<int>, std::vector<uint8_t>> GetData(const std::string &fileName);
          std::tuple<torch::Tensor, torch::Tensor> GetData(const std::string &fileName);
          void ConvertToTensor(std::vector<char> &vec, const int size);
        public:

          // The mode in which the dataset is loaded.
          enum class Mode { kTrain, kTest };
          // Loads the CIFAR10 dataset from the `root` path.
          //
          // The supplied `root` path should contain the *content* of the unzipped
          // CIFAR10 dataset, available from
          // https://www.cs.toronto.edu/~kriz/cifar.html
          // https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
          explicit CIFAR10(const std::string &root, Mode mode = Mode::kTrain);

          virtual ~CIFAR10();

          /// Returns the `Example` at the given `index`.
          Example<> get(size_t index) override;

          /// Returns the size of the dataset.
          optional<size_t> size() const override;

#if 0
          /// Returns true if this is the training subset of CIFAR10.
          bool is_train() const noexcept;

          /// Returns all images stacked into a single tensor.
          const Tensor& images() const;

          /// Returns all targets stacked into a single tensor.
          const Tensor& targets() const;
#endif
      };
    }// ns datasets
  }// ns data
}// ns torch
