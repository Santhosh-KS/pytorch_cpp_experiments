#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <memory>

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Cifar10DataSetParser.hpp"


namespace TDD = torch::data::datasets;

struct ReLu: torch::nn::Module {
  ReLu() {}
  torch::Tensor forward(torch::Tensor x) {
    return torch::relu(x);
  }
};

struct MaxPool2d: torch::nn::Module {
  MaxPool2d() {}
  torch::Tensor forward(torch::Tensor x) {
    return torch::max_pool2d(x,{2, 2});
  }
  void pretty_print(std::ostream& stream) const {
    stream << "torch::max_pool2d(x, {2, 2})";
  }
};

struct Flatten: torch::nn::Module {
  Flatten() {}
  torch::Tensor forward(torch::Tensor x) {
    return x.view({-1, 64*4*4});
  }
};

struct DropOut: torch::nn::Module {
  double Rate;
  bool IsTrain;

  DropOut(double rate, bool train):Rate(rate),IsTrain(train) {}
  torch::Tensor forward(torch::Tensor x) {
    return torch::dropout(x, Rate, IsTrain);
  }

  void pretty_print(std::ostream& stream) const {
    stream << "torch::nn::Dropout(rate=" << Rate << ")";
  }
};

struct LogSoftMax : torch::nn::Module {

  LogSoftMax() {}

  torch::Tensor forward(torch::Tensor x) {
    return torch::log_softmax(x, /*dim=*/1);
  }

  void pretty_print(std::ostream& stream) const {
    stream << "torch::log_softmax(x, dim=1)";
  }
};

bool Display(const torch::Tensor &imageTensor, const std::string &title)
{
  bool retVal(false);
  std::vector<uint8_t> tmpVec;
  torch::Tensor flattenImg = imageTensor.view({3*32*32});

  for(int i =0; i < flattenImg.numel(); i++) {
    uint8_t val = static_cast<uint8_t>(flattenImg[i].item().to<float>()*255);
    tmpVec.push_back(val);
    retVal = true;
  }

#if 0
  // Merge the color channels appropriately.
  cv::Mat outputMat(3, 32*32, CV_8UC1, tmpVec.data());
  cv::Mat tmp = outputMat.t();
  outputMat = tmp.reshape(3, 32);
  cv::cvtColor(outputMat, outputMat, cv::COLOR_RGB2BGR);
  cv::imshow("testing", outputMat);
  cv::waitKey(0);
#endif

  // Merge the color channels appropriately.
  cv::Mat outputMat;

  cv::Mat channelR(32, 32, CV_8UC1, tmpVec.data());
  cv::Mat channelG(32, 32, CV_8UC1, tmpVec.data() + 32 * 32);
  cv::Mat channelB(32, 32, CV_8UC1, tmpVec.data() + 2 * 32 * 32);
  std::vector<cv::Mat> channels{ channelB, channelG, channelR };

  cv::merge(channels, outputMat);
  cv::imshow(title.c_str(), outputMat);
  cv::waitKey(0);
  return retVal;
}

std::stringstream ReadFile(const std::string &file)
{
  std::stringstream buffer;
  std::ifstream fileReader(file);
  if(fileReader) {
    buffer << fileReader.rdbuf();
    fileReader.close();
  }
  return buffer;
}

#if 0
struct TestModule : public Cloneable<TestModule> {
  TestModule() {
    reset();
  }
  void reset() override {
    l1 = register_module("l1", Linear(10, 3));
    l2 = register_module("l2", Linear(3, 5));
    l3 = register_module("l3", Linear(5, 100));
    buffer = register_buffer("buf", torch::ones({2, 2}));
  }

  Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
  torch::Tensor buffer;
};
#endif
//struct CifarNetImpl : torch::nn::Module {
struct CifarNetImpl : public torch::nn::Cloneable<CifarNetImpl> {


  using torch::nn::Module::register_module;

  torch::nn::Conv2d ConvLayer1{nullptr};
  torch::nn::Conv2d ConvLayer2{nullptr};
  torch::nn::Conv2d ConvLayer3{nullptr};
  torch::nn::Linear LinearLayer1{nullptr};
  torch::nn::Linear LinearLayer2{nullptr};

  CifarNetImpl ()
  {
    reset();
  }

  void reset() override {

    ConvLayer1 = register_module("ConvLayer1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
    ConvLayer2 = register_module("ConvLayer2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
    ConvLayer3 = register_module("ConvLayer3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    LinearLayer1 = register_module("LinearLayer1", torch::nn::Linear(64 * 4 * 4, 500));
    LinearLayer2 = register_module("LinearLayer2", torch::nn::Linear(500, 10));
  }

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

//TORCH_MODULE(CifarNet);

int main()
{
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  uint32_t batchSize = 64;

  auto dataSet = torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);

  auto trainDataLoader = torch::data::make_data_loader(
      dataSet.map(torch::data::transforms::Stack<>()),
      batchSize);

#if 0
  // Test if the images loaded properly
  auto batch = std::begin(*trainDataLoader);

  auto images = batch->data;
  auto target = batch->target;
  std::cout << "images = " << images.sizes() << "\n";
  std::cout << "targets = " << target.sizes() << "\n";

  std::cout << "single images = " << images[0].sizes() << "\n";

  // Test if the images are decoded fine.
  for (int i = 0 ; i < images.size(0); i++) {
    std::string title = dataSet.GetTarget(target[i].item<int>());
    std::cout << "Displaying " << (dataSet.IsTrain() ? " Tain " : " Test ") <<"Image # " << i << " is : " << title.c_str() <<" \n";
    Display(images[i], title);
  }
  cv::destroyAllWindows();

#endif
  // Check if GPU is available.
  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  else {
    std::cout << "Training on CPU." << std::endl;
  }

#if 0
  torch::nn::Sequential seqConvLayer(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
      ReLu(),
      MaxPool2d(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
      ReLu(),
      MaxPool2d(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
      ReLu(),
      MaxPool2d(),
      Flatten(),
      DropOut(0.25, true),
      torch::nn::Linear(64 * 4 * 4, 500),
      ReLu(),
      DropOut(0.25, true),
      torch::nn::Linear(500, 10),
      LogSoftMax()
      );
#endif
  //CifarNet seqConvLayer(0.25);
  //auto seqConvLayer = std::make_shared<CifarNet>();
//  CifarNet seqConvLayer;
  CifarNetImpl seqConvLayer;
  //seqConvLayer->to(device);
  seqConvLayer.to(device);
  std::cout << "Model:\n\n";
  std::cout << c10::str(seqConvLayer) << "\n\n";

  //torch::optim::SGD optimizer(seqConvLayer->parameters(), /*lr=*/0.01);
  torch::optim::SGD optimizer(seqConvLayer.parameters(), /*lr=*/0.01);

  std::cout << "Training.....\n";

#if 0
  double minVal(10000.991);
  for (size_t epoch = 1; epoch <= 2; ++epoch) {

    size_t batchIndex = 0;
    // keep track of training and validation loss
    float train_loss = 0.0;
    //float  valid_loss = 0.0;

    //seqConvLayer->train();
    seqConvLayer.train();
    for (auto& batch : *trainDataLoader) {

      batch.data.to(device);
      batch.target.to(device);

      // Reset gradients.
      optimizer.zero_grad();

      // Execute the model on the input data.
      auto imgs = batch.data.to(torch::kFloat);
      //std::cout << "images = " << imgs.sizes() << "\n";


      // forward pass: compute predicted outputs by passing inputs to the model
      //torch::Tensor prediction = seqConvLayer->forward(imgs);
      torch::Tensor prediction = seqConvLayer.forward(imgs);

      // calculate the batch loss
      auto loss = torch::nll_loss(prediction, batch.target.to(torch::kLong));
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();

      // Update the parameters based on the calculated gradients.
      optimizer.step();

      // update training loss
      train_loss += loss.item<float>() * batch.data.size(0);

      // Output the loss and checkpoint every 100 batches.
      if (++batchIndex % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batchIndex
          << " | Training Loss: " << loss.item<float>() << "\n";
      }
      if (minVal >  loss.item<float>()) {
        minVal =  loss.item<float>();
#if 0
        std::string model_path = "new_test.pt";
        torch::save(seqConvLayer, model_path);
#endif

        std::cout << "Saving model with least training error = " << minVal << "\n";
      }
    }
  }
  std::cout << "Least training error reached = " << minVal << "\n";

  torch::serialize::InputArchive archive;

  std::string file = "new_test.pt";
  // torch::nn::Sequential savedSeq;
  CifarNet newCifarnet;
  //torch::load(seqConvLayer, file);
  torch::load(newCifarnet, file);

  //for (auto& p : seqConvLayer->named_parameters()) {
  for (auto& p : newCifarnet.named_parameters()) {
    std::cout << p.key() << std::endl;
    // Access value.
    std::cout << p.value() << std::endl;
  }

  std::cout << "Saved Model:\n\n";
  std::cout << c10::str(newCifarnet) << "\n\n";
#endif
  return 0;
}
