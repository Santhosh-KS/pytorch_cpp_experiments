#include <iostream>
#include <algorithm>

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Cifar10DataSetParser.hpp"

namespace TDD = torch::data::datasets;

struct ReLu: torch::nn::Module {
  ReLu() {}
  torch::Tensor forward(torch::Tensor x) {
    return torch::relu(x);
  }
};

struct LogSoftMax : torch::nn::Module {
  LogSoftMax() {}
  torch::Tensor forward(torch::Tensor x) {
    return torch::log_softmax(x, /*dim=*/1);
  }
};

bool Display(torch::Tensor imageTensor)
{
  bool retVal(false);
  std::vector<uint8_t> tmpVec;
  auto flattenImg = imageTensor.view({3*32*32});

#if 0
  for(int i = 0; i < flattenImg.numel(); i++) {
    std::cout << flattenImg[i] << " " ;
  }
  std::cout << "Done pinting vector \n\n";
#endif
  for(int i =0; i < flattenImg.numel(); i++) {
    tmpVec.push_back(flattenImg[i].item().to<float>()*255);
    retVal = true;
  }

  auto minMax = std::minmax_element (tmpVec.begin(), tmpVec.end());
  std::cout << "Tmp vec = " << tmpVec.size() << " Min = " << static_cast<int>(*minMax.first) << " Max = " << static_cast<int>(*minMax.second) << "\n";
#if 0
  for(size_t i = 0; i < tmpVec.size(); i++) {
    std::cout << tmpVec[i] << " ";
  }
  std::cout << "\n";
#endif
  cv::Mat imgMat;
  imgMat.create(32, 32,  CV_8UC3);
  memcpy(imgMat.data, tmpVec.data(), tmpVec.size()*sizeof(uint8_t));
  cv::imshow("testing", imgMat);
  cv::waitKey(0);
  return retVal;
}

int main()
{
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);
#if 0
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  uint32_t batchSize = 64;
  auto trainDataLoader = torch::data::make_data_loader(
      torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode).map(
        torch::data::transforms::Stack<>()),
      batchSize);
  auto batch = std::begin(*trainDataLoader);

  auto images = batch->data;
  auto target = batch->target;
  std::cout << "images = " << images.sizes() << "\n";
  std::cout << "targets = " << target.sizes() << "\n";

  std::cout << "single images = " << images[0].sizes() << "\n";

  for (int i = 0 ; i < images.size(0); i++) {
    std::cout << "Image # " << i << " ";
    Display(images[i]);
  }


  torch::nn::Sequential sequential(torch::nn::Linear(3072, 1024),
      //torch::nn::Functional(torch::relu),
      ReLu(),
      torch::nn::Linear(1024, 512),
      ReLu(),
      torch::nn::Linear(512, 256),
      ReLu(),
      torch::nn::Linear(256, 128),
      ReLu(),
      torch::nn::Linear(128, 64),
      //torch::nn::Functional(torch::relu),
      ReLu(),
      torch::nn::Linear(64, 10),
      LogSoftMax());

  std::cout << "Model:\n\n";
  std::cout << c10::str(sequential) << "\n\n";

  torch::optim::SGD optimizer(sequential->parameters(), /*lr=*/0.01);

  std::cout << "Training:\n\n";
  for (size_t epoch = 1; epoch <= 6; ++epoch) {
    size_t batchIndex = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *trainDataLoader) {

      // Reset gradients.
      optimizer.zero_grad();

      // Execute the model on the input data.
      auto imgs = batch.data.view({batch.data.size(0), -1});

      torch::Tensor prediction = sequential->forward(imgs);

      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);

      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();

      // Update the parameters based on the calculated gradients.
      optimizer.step();

      // Output the loss and checkpoint every 100 batches.
      if (++batchIndex % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batchIndex
          << " | Training Loss: " << loss.item<float>() << "\n\n";
      }
    }
  }
#endif

  return 0;
}
