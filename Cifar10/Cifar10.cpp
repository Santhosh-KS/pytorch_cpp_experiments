#include <iostream>

#include <torch/torch.h>

#include "Cifar10DataSetParser.hpp"

namespace TDD = torch::data::datasets;
int main()
{
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  auto train = TDD::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);
  mode = TDD::CIFAR10::Mode::kTest;
  auto test = TDD::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);
#if 0
  uint32_t batchSize = 64;
  auto trainDataLoader = torch::data::make_data_loader(
      torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/").map(
        torch::data::transforms::Stack<>()),
      batchSize);
#endif
  return 0;
}
