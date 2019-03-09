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
  torch::manual_seed(9);

  torch::Tensor features = torch::rand({1, 3});
  //auto features = torch::rand({1, 3});
  std::cout << "features = " << features << "\n";

  // Define the size of each layer in our network
  // Number of input units, must match number of input features
  auto num_inputs = features.size(1);
  // Number of hidden units
  int num_hidden_layers = 2;
  // Number of output units
  int num_output = 1;


  // Weights for inputs to hidden layer
  auto W1 = torch::randn({num_inputs, num_hidden_layers});
  std::cout << "W1 = " << W1.sizes() << "\n";
  // Weights for hidden layer to output layer
  auto W2 = torch::randn({num_hidden_layers, num_output});
  std::cout << "W2 = " << W2.sizes() << "\n";

  // and bias terms for hidden and output layers
  auto B1 = torch::randn({1, num_hidden_layers});
  std::cout << "B1 = " << B1.sizes() << "\n";
  auto B2 = torch::randn({1, num_output});
  std::cout << "B2 = " << B2.sizes() << "\n";


  auto h = ActivationFunction(torch::mm(features, W1) + B1);
  auto y = ActivationFunction(torch::mm(h, W2) + B2);
  std::cout << "y = " << y << "\n";
  return 0;
}
