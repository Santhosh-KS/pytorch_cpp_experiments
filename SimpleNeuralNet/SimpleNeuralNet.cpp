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
//  torch::manual_seed(9);
  //at::manual_seed(9);

  torch::Tensor features = torch::rand({1, 5});
  std::cout << "features = " << features << "\n";

  auto weights = torch::randn_like(features);
  std::cout << "weights = " << weights << "\n";

  auto bias = torch::randn({1,1});
  std::cout << "bias  =" << bias << "\n";

  // There are multiple ways to get the same result. Here are the few of them.
  // 1st way
  auto y = ActivationFunction(torch::sum(features * weights) +bias);
  std::cout << "y using (\"torch::sum()\") = " << y << "\n";

  // 2nd way
  y = ActivationFunction((features * weights).sum() +bias);
  std::cout << "y using (\".sum()\") = " << y << "\n";

  // 3rd and preferred way. Using Matrix multiplication.
  y = ActivationFunction(torch::mm(features, weights.view({5,1}))+bias);
  std::cout << "y using (\"torch::mm().view()\") = " << y << "\n";

  // Reshaping the tensor using reshape().
  y = ActivationFunction(torch::mm(features, weights.reshape({5,1}))+bias);
  std::cout << "y using (\"torch.mm().reshape() \") = " << y << "\n";

  // Reshaping the tensor using inplace resize_().
  y = ActivationFunction(torch::mm(features, weights.resize_({5,1}))+bias);
  std::cout << "y using (\"torch.mm().resize_() \") = " << y << "\n";


  std::cout << "features shape = " << features.sizes() << "\n";
  std::cout << "features shape rows = " << features.sizes()[0] << "\n";
  std::cout << "features shape col = " << features.size(1) << "\n";
  std::cout << "features shape dtype = " << features.dtype() << "\n";
  std::cout << "features shape numel = " << features.numel() << "\n";

  auto rInt = torch::randint(64, {1,10});
  std::cout << "rInt = " << rInt << "\n";
  return 0;
}
