#include <torch/torch.h>
#include <iostream>
//#include <memory>


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
  auto model = std::make_unique<NetWork>();
  auto parameters = model->named_parameters();
  auto keys = parameters.keys();
  auto vals = parameters.values();
#if 1
  for(auto v: keys) {
    std::cout << v << "\n";
  }
  std::cout << "Hidden layer weights = " << parameters["hidden.weight"].sizes() << "\n";
  std::cout << "Hidden layer bias = " << parameters["hidden.bias"].sizes() << "\n";
  std::cout << "Output layer weights = " << parameters["output.weight"].sizes() << "\n";
  std::cout << "Output layer bias = " << parameters["output.bias"].sizes() << "\n";
#endif
#if 0
  for(auto v: vals) {
    std::cout << v << "\n";
  }
#endif

  torch::nn::Sequential seqDecType1(torch::nn::Linear(784, 128),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(128, 64),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(64, 10),
      LogSoftMax());

  std::cout << c10::str(seqDecType1) << "\n";

  torch::nn::Sequential seqDecType2(torch::nn::Linear(784, 128),
      ReLu(),
      torch::nn::Linear(128, 64),
      ReLu(),
      torch::nn::Linear(64, 10),
      LogSoftMax());

  std::cout << c10::str(seqDecType2) << "\n";

  torch::optim::SGD optimizer(seqDecType1->parameters(), /*lr=*/0.01);
  return 0;
}
