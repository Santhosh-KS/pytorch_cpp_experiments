#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

namespace TDT = torch::data::transforms;

int main()
{
  //torch::Tensor tensor = torch::rand({2, 3});
  //std::cout << tensor << std::endl;
  //TDT::Compose();
  //https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth
  //resnet18-5c106cde.pth

  //auto sequential = std::make_shared<torch::nn::Sequential>();
  //torch::nn::Sequential sequential;

  //torch::load(sequential, "resnet18-5c106cde.pth");

  std::shared_ptr<torch::jit::script::Module> module;
  module = torch::jit::load("script_module.pt");
  std::cout << c10::str(module) << "\n";

//  std::cout << c10::str(sequential) << "\n";
}
