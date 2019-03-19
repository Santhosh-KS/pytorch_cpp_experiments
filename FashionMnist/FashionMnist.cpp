#include <iostream>
#include <string>

#include <torch/torch.h>

#include "FashionMnistData.hpp"

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

std::string Item(const int id)
{
  std::string retVal("None");
  switch(id) {
    case 0:
      retVal = "T-shirt/top";
      break;
    case 1:
      retVal = "Trouser";
      break;
    case 2:
      retVal = "Pullover";
      break;
    case 3:
      retVal = "Dress";
      break;
    case 4:
      retVal = "Coat";
      break;
    case 5:
      retVal = "Sandal";
      break;
    case 6:
      retVal = "Shirt";
      break;
    case 7:
      retVal = "Sneaker";
      break;
    case 8:
      retVal = "Bag";
      break;
    case 9:
      retVal = "Ankle boot";
      break;
    default:
      break;
  }
  return retVal;
}

int main()
{
  // Train model.
  uint32_t batchSize = 64;
  auto trainDataLoader = torch::data::make_data_loader(
      torch::data::datasets::FASHION_MNIST("/opt/FASHION_MNIST/").map(
        torch::data::transforms::Stack<>()),
      batchSize);

  torch::nn::Sequential sequential(torch::nn::Linear(784, 128),
      //torch::nn::Functional(torch::relu),
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

  // Test the built model.

  sequential->eval();
  std::cout << "Testing:\n\n";
  torch::data::datasets::FASHION_MNIST::Mode  mode = torch::data::datasets::FASHION_MNIST::Mode::kTest;
  auto testDataLoader = torch::data::make_data_loader(
      torch::data::datasets::FASHION_MNIST("/opt/FASHION_MNIST/", mode).map(
        torch::data::transforms::Stack<>()),
      batchSize);

  auto batch = std::begin(*testDataLoader);

  auto images = batch->data;
  auto target = batch->target;
  //  std::cout << "images = " << images.sizes() << "\n";
  //  std::cout << "targets = " << target.sizes() << "\n";

  auto index = torch::randint(batchSize,{1,batchSize}, at::kInt);

  std::cout << "+---------------+---------------+-------------+\n";
  std::cout << "|  Actual value |   Prediction  |   Accuracy  |\n";
  std::cout << "|---------------|---------------|-------------|\n";

  for(uint32_t i = 0; i < batchSize; i++) {

    //let us predict the image results from our model.
    auto image = images[i];
    //std::cout << "image = " << image.sizes() << "\n";

    auto img = image.view({1,784});
    //std::cout << "img = " << img.sizes() << "\n";
    // std::cout << img << "\n";
    auto logProb = sequential->forward(img);
    //auto result = std::get<1>(sequential(img).max(/*dim=*/1));

    auto prediction = torch::exp(logProb);
    //std::cout << "prediction = " << prediction << "\n";
    auto maxVal = prediction.max(1);
    std::cout << "|  " << Item(std::get<1>(maxVal).item<int>()).c_str() <<  "  |  " << Item(target[i].item<int>()).c_str() << "  |  "<< std::get<0>(maxVal).item<float>() << "   |\n";
    std::cout << "+---------------+---------------+-------------+\n";
  }
  return 0;
}
