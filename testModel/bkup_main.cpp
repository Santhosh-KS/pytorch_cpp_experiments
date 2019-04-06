#include <iostream>
#include <memory>

#include <torch/torch.h>
//#include <torch/nn/module.h>
//#include <torch/nn/modules/conv.h>


namespace TDD = torch::data::datasets;

struct TestNetImpl : torch::nn::Module {
  //struct TestNetImpl : public torch::nn::Cloneable<TestNetImpl> {

  torch::nn::Conv2d ConvLayer1{nullptr};
  torch::nn::Conv2d ConvLayer2{nullptr};
  torch::nn::Conv2d ConvLayer3{nullptr};
  torch::nn::Linear LinearLayer1{nullptr};
  torch::nn::Linear LinearLayer2{nullptr};

  TestNetImpl() :
    ConvLayer1(register_module("ConvLayer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)))),
    ConvLayer2(register_module("ConvLayer2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)))),
    ConvLayer3(register_module("ConvLayer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)))),
    LinearLayer1(register_module("LinearLayer1", torch::nn::Linear(64 * 4 * 4, 500))),
    LinearLayer2(register_module("LinearLayer2", torch::nn::Linear(500, 10)))
  {
    // Empty.
  }

#if 0
  TestNetImpl ()
  {
    reset();
  }
  void reset() override {
    using torch::nn::Module::register_module;

    ConvLayer1 = register_module("ConvLayer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
    ConvLayer2 = register_module("ConvLayer2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
    ConvLayer3 = register_module("ConvLayer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    LinearLayer1 = register_module("LinearLayer1", torch::nn::Linear(64 * 4 * 4, 500));
    LinearLayer2 = register_module("LinearLayer2", torch::nn::Linear(500, 10));
  }
#endif

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(torch::max_pool2d(ConvLayer1->forward(x), 2));
    x = torch::relu(torch::max_pool2d(ConvLayer2->forward(x), 2));
    x = torch::relu(torch::max_pool2d(ConvLayer3->forward(x), 2));

    // Flatten the image tensor appropiately
    x = x.view({-1, 64*4*4});
    x = torch::dropout(x, 0.25, is_training());
    x = torch::relu(LinearLayer1->forward(x));
    x = torch::dropout(x, 0.25, is_training());
    x = torch::relu(LinearLayer2->forward(x));
    return torch::log_softmax(x, /*dim=*/1);
  }
};

TORCH_MODULE(TestNet);

int main()
{
  // Check if GPU is available.
  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  else {
    std::cout << "Training on CPU." << std::endl;
  }

  TestNet model;
  model.to(device);

  torch::optim::SGD optimizer(model.parameters(), /*lr=*/0.01);

  return 0;
}
