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

  for(int i =0; i < flattenImg.numel(); i++) {
    tmpVec.push_back(static_cast<uint8_t>(flattenImg[i].item().to<float>()*255));
    retVal = true;
  }

#if 0
  // Merge the color channels appropriately.
  cv::Mat outputMat(3, 32*32, CV_8UC1, tmpVec.data());
  cv::Mat tmp = outputMat.t();
  outputMat = tmp.reshape(3, 32);
  cv::cvtColor(outputMat, outputMat, cv::COLOR_RGB2BGR);
  cv::imshow("testing", outputMat);
  cv::waitKey(0);
#endif

  // Merge the color channels appropriately.
  cv::Mat outputMat;

  cv::Mat channelR(32, 32, CV_8UC1, tmpVec.data());
  cv::Mat channelG(32, 32, CV_8UC1, tmpVec.data() + 32 * 32);
  cv::Mat channelB(32, 32, CV_8UC1, tmpVec.data() + 2 * 32 * 32);
  std::vector<cv::Mat> channels{ channelB, channelG, channelR };

  cv::merge(channels, outputMat);
  cv::imshow("testing", outputMat);
  cv::waitKey(0);

  return retVal;
}

int main()
{
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  uint32_t batchSize = 64;

  auto dataSet = torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);

  auto trainDataLoader = torch::data::make_data_loader(
      dataSet.map(torch::data::transforms::Stack<>()),
      batchSize);
  // Test the image loader
#if 1
  auto batch = std::begin(*trainDataLoader);

  auto images = batch->data;
  auto target = batch->target;
  std::cout << "images = " << images.sizes() << "\n";
  std::cout << "targets = " << target.sizes() << "\n";

  std::cout << "single images = " << images[0].sizes() << "\n";

  // Test if the images are decoded fine.
  for (int i = 0 ; i < images.size(0); i++) {
    std::cout << "Image # " << i << " is : " << dataSet.GetTarget(target[i].item<int>()).c_str() <<" \n";
    Display(images[i]);
  }
#endif

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

#if 0
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
