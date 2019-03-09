#include <torch/torch.h>
#include <ATen/Context.h>
#include <iostream>

torch::Tensor ActivationFunction(const torch::Tensor &x)
{
  // Sigmoid function.
  auto retVal = 1/(1+torch::exp(-x));
  return retVal;
}

int main()
{
  // Seed for Random number
  at::manual_seed(9);

  torch::Tensor features = torch::rand({1, 5});
  std::cout << "features = " << features << "\n";

  auto weights = torch::randn_like(features);
  std::cout << "weights = " << weights << "\n";

  auto bias = torch::randn({1,1});
  std::cout << "bias  =" << bias << "\n";

  auto y = ActivationFunction(torch::sum(features * weights) +bias);
  std::cout << "y = " << y << "\n";

  y = ActivationFunction(torch::mm(features, weights.view({5,1}))+bias);
  std::cout << "y = " << y << "\n";
}
