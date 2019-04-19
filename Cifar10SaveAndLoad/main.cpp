#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <memory>

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>

#include "Cifar10DataSetParser.hpp"


#if OPENCV_INCLUDED
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


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
#endif

namespace TDD = torch::data::datasets;

struct CifarNetImpl : torch::nn::Cloneable<CifarNetImpl> {

  using torch::nn::Module::register_module;

  double Rate;

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

TORCH_MODULE(CifarNet);


torch::Device GetDevice()
{
  // Check if GPU is available.
  torch::Device device(torch::kCPU);

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!\n";
    device = torch::Device(torch::kCUDA);
  }
  else {
    std::cout << "Let us setlle for a CPU.";
  }
  return device;
}

void TrainModel(const std::string &file)
{
  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTrain;
  uint32_t batchSize = 64;

  auto dataSet = torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);

  auto trainDataLoader = torch::data::make_data_loader(
      dataSet.map(torch::data::transforms::Stack<>()),
      batchSize);

#if OPENCV_INCLUDED
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

  torch::Device device(GetDevice());

  CifarNet model;
  model->to(device);
  std::cout << "Model:\n\n";
  std::cout << c10::str(model) << "\n\n";

  torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.075);

  std::cout << "Training.....\n";

  double minVal(10000.991);
  //for (size_t epoch = 1; epoch <= 30; ++epoch) {
  for (size_t epoch = 1; epoch <= 2; ++epoch) {

    size_t batchIndex = 0;
    // keep track of training and validation loss
    float train_loss = 0.0;

    model->train();
    for (auto& batch : *trainDataLoader) {

      batch.data.to(device);
      batch.target.to(device);

      // Reset gradients.
      optimizer.zero_grad();

      // Execute the model on the input data.
      auto imgs = batch.data.to(torch::kFloat);
      //std::cout << "images = " << imgs.sizes() << "\n";


      // forward pass: compute predicted outputs by passing inputs to the model
      torch::Tensor prediction = model->forward(imgs);

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
        torch::save(model, file);
        std::cout << "Saving model with least training error = " << minVal << "\n";
      }
    }
  }
  std::cout << "Least training error reached = " << minVal << "\n";
}

void TestModel(const std::string &file)
{
  CifarNet model;
  torch::load(model, file);
#if 0
  for (auto& p : model->named_parameters()) {
    std::cout << p.key() << std::endl;
    // Access value.
    std::cout << p.value() << std::endl;
  }
#endif
  if (model->named_parameters().size() <= 0) {
    std::cout << file.c_str() << "loading failed\n";
    return;
  }
  std::cout << "Saved Model:\n\n";
  std::cout << c10::str(model) << "\n\n";

  torch::Device device(GetDevice());
  model->to(device);

  TDD::CIFAR10::Mode  mode = TDD::CIFAR10::Mode::kTest;
  uint32_t batchSize = 64;

  auto dataSet = torch::data::datasets::CIFAR10("/opt/pytorch/data/cifar-10-batches-bin/", mode);

  auto testDataLoader = torch::data::make_data_loader(
      dataSet.map(torch::data::transforms::Stack<>()),
      batchSize);
#if 0
  auto batch = std::begin(*testDataLoader);
  auto images = batch->data;
  auto target = batch->target;

  images.to(device);
  target.to(device);
#endif
  model->eval();

  std::vector<int> actualData(10,0);
  std::vector<int> predictedData(10,0);

  for (auto& batch : *testDataLoader) {

      batch.data.to(device);
      batch.target.to(device);
    // Execute the model on the input data.
    auto img = batch.data.to(torch::kFloat);

    //std::cout << "Image size = " << img.sizes() << "\n";
    // LogProbability.
    auto logProb = model->forward(img);
    auto prediction = torch::exp(logProb);
    auto maxVal = prediction.max(1);

    auto itemSize = std::get<1>(maxVal).sizes();


    for (uint32_t i = 0; i < itemSize[0]; i++) {
      int tmpPredict = std::get<1>(maxVal)[i].item<int>();
      int tmpActual = batch.target[i].item<int>();
      //std::cout << i << " Prediction = " << tmpPredict << " Actual val = " << tmpActual <<  "\n";
      //std::cout << i << " Prediction = " << dataSet.GetTarget(tmpPredict).c_str() << " Actual val = " << dataSet.GetTarget(tmpActual).c_str() <<  "\n";

      if (tmpActual == tmpPredict) {
        predictedData[tmpPredict] += 1;
      }
      actualData[tmpActual] += 1;
    }
  }

  for(int i = 0; i < 10; i++) {
    //std::cout << dataSet.GetTarget(i) << " Total actual = " << actualData[i] << " Predicted = " << predictedData[i] << "\n";

    double percent  = double(predictedData[i])/double(actualData[i]);
    std::cout << dataSet.GetTarget(i) << std::setprecision (4) <<" " << percent << "\n";

  }
}

void PrintUsage()
{
  std::cout<< "USAGE : ./Cifar10SaveLoad <0/1>\n";
  std::cout<< "0 : Train \n1 : Test\n";
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    PrintUsage();
    return 0;
  }

  int option = std::stoi(argv[1]);
  if (option > 1 || option < 0) {
    PrintUsage();
    return 0;
  }
//  std::string modelName("new_test.pt");
  std::string modelName("BestModel.pt");
  option ? TestModel(modelName) : TrainModel(modelName);
  return 0;
}
