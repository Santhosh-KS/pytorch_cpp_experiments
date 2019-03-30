#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/script.h>

#include "caffe2/serialize/inline_container.h"

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

  std::shared_ptr<torch::jit::script::Module> module =  torch::jit::load("script_module.pt");
  std::cout << c10::str(module) << "\n";

#if 0
  for (const auto& parameter : module->get_parameters()) {
    std::cout << parameter->slot() << "\n";
  }

  //caffe2::serialize::PyTorchStreamReader pytorchSerializer("script_module.pt");

//  std::cout << pytorchSerializer.archiveName().c_str() << "\n";

 // auto sequential = std::make_shared<torch::nn::Sequential>();
  torch::nn::Sequential sequential;

  torch::load(sequential, "script_module.pt");

  std::cout << c10::str(sequential) << "\n";
  std::cout << sequential->size() << "\n";


  auto ret = module->get_modules();
  auto vals = ret.values();

  for (auto &v: vals) {
    //if (*(module->get_method(v.name).get())) {
      std::cout << module->get_method(v.name).pretty_print_schema().c_str() << "\n";
    //}
  }

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  auto ret = module->get_modules();

  auto keys = ret.keys();
  auto vals = ret.values();

  for (auto &v: keys) {
    std::cout << v.c_str() << "\n";
  }
  for (auto &v: vals) {
    //std::cout << module->get_method(v.name) << "\n";
    std::cout << v.name.c_str() << "\n";
  }


  torch::nn::Module mdl(module->forward(inputs));
  torch::nn::Sequential sequential(mdl);
  std::cout << c10::str(sequential) << "\n";

  auto parameters = module.get_parameters();
  auto keys = parameters.keys();
  auto vals = parameters.values();
  for(auto v: keys) {
    std::cout << v << "\n";
  }
#endif
  return 0;
  //  std::cout << c10::str(sequential) << "\n";
}
