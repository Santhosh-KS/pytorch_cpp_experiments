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
  auto trainDataLoader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/opt/MNIST/").map(
        torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  torch::nn::Sequential sequential(torch::nn::Linear(784, 128),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(128, 64),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(64, 10),
      LogSoftMax());

  std::cout << c10::str(sequential) << "\n";

  torch::optim::SGD optimizer(sequential->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *trainDataLoader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      auto imgs = batch.data.view({batch.data.size(0), -1});
      //torch::Tensor prediction = sequential->forward(batch.data);
      torch::Tensor prediction = sequential->forward(imgs);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
          << " | Training Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(sequential, "sequential.pt");
      }
    }
  }
//  torch::nn::Sequential loadSeq;
//  torch::load(loadSeq, "sequential.pt");
//  std::cout << "Loaded model = " << c10::str(loadSeq) << "\n";
  return 0;
}
