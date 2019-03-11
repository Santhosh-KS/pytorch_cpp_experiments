#include <torch/torch.h>
#include <ATen/Context.h>
#include <iostream>

torch::Tensor ActivationFunction(const torch::Tensor &x)
{
  // Sigmoid function.
  auto retVal = 1/(1+torch::exp(-x));
  return retVal;
}

torch::Tensor SoftMax(const torch::Tensor &x)
{
  auto sum = torch::sum(torch::exp(x),1);
  return torch::exp(x)/sum.view({-1, 1});
}

int main()
{

  // Download the MNIST data using the script present in,
  // ../scripts/download_mnist.py
  // Create a data loader for the MNIST dataset.
  auto trainLoader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("/opt/MNIST/").map(
        torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  auto batch = std::begin(*trainLoader);

  auto images = batch->data;
  auto target = batch->target;
  std::cout << "images = " << images.sizes() << "\n";
  std::cout << "targets = " << target.sizes() << "\n";

  // Seed for Random number
  torch::manual_seed(9);

  auto inputs = images.view({images.size(0),-1});
  std::cout << "inputs = " << inputs.sizes() << "\n";


  // Create parameters
  auto w1 = torch::randn({784, 256});
  std::cout << "w1 =" << w1.sizes() << "\n";
  auto b1 = torch::randn({256});
  std::cout << "b1 =" << b1.sizes() << "\n";

  auto w2 = torch::randn({256, 10});
  std::cout << "w2 =" << w2.sizes() << "\n";
  auto b2 = torch::randn({10});
  std::cout << "b2 =" << b2.sizes() << "\n";

  auto h = ActivationFunction(torch::mm(inputs, w1) + b1);
  std::cout << "h =" << h.sizes() << "\n";
  auto out = torch::mm(h, w2) + b2;
  std::cout << "out =" << out.sizes() << "\n";
  //std::cout << "out =" << out << "\n";

  auto pred = SoftMax(out);

  std::cout << "pred = " << pred.sizes() << "\n";
  //  std::cout << "pred = " << pred << "\n";

  auto predSum = torch::sum(pred, 1);
  std::cout << "predsum shape = " << predSum.sizes() << "\n";
  //std::cout << "predsum = " << predSum << "\n";
  return 0;
}
