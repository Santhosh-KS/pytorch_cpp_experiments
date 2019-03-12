#include <torch/torch.h>
#include <iostream>


class NetWork : public torch::nn::Module
{
  public:
    torch::nn::Linear HiddenLayer;
    torch::nn::Linear OutputLayer;
    NetWork();
    ~NetWork();
    torch::Tensor Forward(torch::Tensor x);
};

NetWork::NetWork():
  HiddenLayer(register_module("hidden", torch::nn::Linear(784, 256))),
  OutputLayer(register_module("output", torch::nn::Linear(256,10)))
{
  // Empty
}

NetWork::~NetWork()
{
  // Empty
}

torch::Tensor NetWork::Forward(torch::Tensor x)
{
  x = HiddenLayer(x);
  x = torch::sigmoid(x);
  x = OutputLayer(x);
  x = torch::softmax(x,/*dim=*/1);
  return x;
}

int main()
{
  auto model = std::make_unique<NetWork>();
  auto parameters = model->named_parameters();
  auto keys = parameters.keys();
  auto vals = parameters.values();
  for(auto v: keys) {
    std::cout << v << "\n";
  }
  std::cout << "Hidden layer weights = " << parameters["hidden.weight"].sizes() << "\n";
  std::cout << "Hidden layer bias = " << parameters["hidden.bias"].sizes() << "\n";
  std::cout << "Output layer weights = " << parameters["output.weight"].sizes() << "\n";
  std::cout << "Output layer bias = " << parameters["output.bias"].sizes() << "\n";
  //  for(auto v: vals) {
  //    std::cout << v << "\n";
  //  }
  return 0;
}
