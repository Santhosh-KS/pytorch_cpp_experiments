#include <torch/torch.h>
#include <iostream>

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

int main()
{
  // Train model.
  uint32_t batchSize = 64;
  auto trainDataLoader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/opt/MNIST/").map(
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

  //std::cout << c10::str(sequential->parameters()) << "\n\n";
  std::cout << "Training:\n\n";
  //for (size_t epoch = 1; epoch <= 6; ++epoch) {
  for (size_t epoch = 1; epoch <= 1; ++epoch) {
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
        torch::save(sequential, "test.pt");
      }
    }
  }
  torch::nn::Sequential test;
  torch::load(test, "test.pt");
  std::cout << "size = " << test->size() << "\n";
  return 0;
}
