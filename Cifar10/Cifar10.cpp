#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <torch/torch.h>
#include <torch/ordered_dict.h>
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
  seqConvLayer->to(device);
  std::cout << "Model:\n\n";
  std::cout << c10::str(seqConvLayer) << "\n\n";

  torch::optim::SGD optimizer(seqConvLayer->parameters(), /*lr=*/0.01);

  std::cout << "Training.....\n";

  double minVal(10000.991);
  for (size_t epoch = 1; epoch <= 2; ++epoch) {

    size_t batchIndex = 0;
    // keep track of training and validation loss
    float train_loss = 0.0;
    //float  valid_loss = 0.0;

    seqConvLayer->train();
    for (auto& batch : *trainDataLoader) {

      batch.data.to(device);
      batch.target.to(device);

      // Reset gradients.
      optimizer.zero_grad();

      // Execute the model on the input data.
      auto imgs = batch.data.to(torch::kFloat);
      //std::cout << "images = " << imgs.sizes() << "\n";


      // forward pass: compute predicted outputs by passing inputs to the model
      torch::Tensor prediction = seqConvLayer->forward(imgs);

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

        std::string model_path = "test_model.pt";
        torch::serialize::OutputArchive output_archive;
        seqConvLayer->save(output_archive);
        output_archive.save_to(model_path);
        std::cout << "Saving model with least training error = " << minVal << "\n";
      }
    }
  }
  std::cout << "Least training error reached = " << minVal << "\n";

  torch::serialize::InputArchive archive;
  std::string file("test_model.pt");
  archive.load_from(file, device);
  //std::stringstream fileContents(ReadFile(file));
  //torch::load(archive, fileContents);
  torch::nn::Sequential savedSeq;

  savedSeq->load(archive);
  auto parameters = savedSeq->named_parameters();
  auto keys = parameters.keys();
  auto vals = parameters.values();

  for(auto v: keys) {
    std::cout << v << "\n";
  }

  std::cout << "Saved Model:\n\n";
  std::cout << c10::str(savedSeq) << "\n\n";
  return 0;
}
