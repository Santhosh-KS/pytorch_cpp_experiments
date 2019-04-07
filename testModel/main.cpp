#include<iostream>

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>


struct TestModuleImpl : public torch::nn::Cloneable<TestModuleImpl> {

  using torch::nn::Module::register_module;
  torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};

#if 0
  TestModuleImpl() :
    l1(register_module("l1", torch::nn::Linear(10, 3))),
    l2(register_module("l2", torch::nn::Linear(3, 5))),
    l3(register_module("l3", torch::nn::Linear(5, 100)))
  {
    // Empty
  }

#endif
  TestModuleImpl() {
    reset();
  }
  void reset() override {
    l1 = register_module("l1", torch::nn::Linear(10, 3));
    l2 = register_module("l2", torch::nn::Linear(3, 5));
    l3 = register_module("l3", torch::nn::Linear(5, 100));
  }
};

TORCH_MODULE(TestModule);

int main()
{
  //TestModule m;
  auto m = std::make_shared<TestModule>();
//  TestModuleImpl m;
  torch::Device device(torch::kCUDA, 0);

  m->to(device);
  m->parameters();
  return 0;
}
