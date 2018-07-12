#include "glow/Importer/TFLite.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/IR/IR.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <fstream>
#include <type_traits>
#include <vector>

using namespace std;
using namespace glow;

const size_t mnistNumImages = 50000;

namespace {
llvm::cl::OptionCategory tfliteCat("TFLite importer Options");
llvm::cl::opt<BackendKind> executionBackend(
    llvm::cl::desc("Backend to use:"),
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter (default option)"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(tfliteCat));
llvm::cl::opt<string> modelFile(
    "m",
    llvm::cl::desc(
        "Specify the file to import TFLite models in Flatbuffer format"),
    llvm::cl::value_desc("file.tflite"), llvm::cl::cat(tfliteCat));
} // namespace

unsigned loadMNIST(Tensor &imageInputs, Tensor &labelInputs) {
  /// Load the MNIST database into 4D tensor of images and 2D tensor of labels.
  imageInputs.reset(ElemKind::FloatTy, {50000u, 28, 28, 1});
  labelInputs.reset(ElemKind::IndexTy, {50000u, 1});

  std::ifstream imgInput("mnist_images.bin", std::ios::binary);
  std::ifstream labInput("mnist_labels.bin", std::ios::binary);

  if (!imgInput.is_open()) {
    llvm::errs() << "Error loading mnist_images.bin\n";
    std::exit(EXIT_FAILURE);
  }
  if (!labInput.is_open()) {
    llvm::errs() << "Error loading mnist_labels.bin\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<char> images((std::istreambuf_iterator<char>(imgInput)),
                           (std::istreambuf_iterator<char>()));
  std::vector<char> labels((std::istreambuf_iterator<char>(labInput)),
                           (std::istreambuf_iterator<char>()));
  float *imagesAsFloatPtr = reinterpret_cast<float *>(&images[0]);

  GLOW_ASSERT(labels.size() * 28 * 28 * sizeof(float) == images.size() &&
              "The size of the image buffer does not match the labels vector");

  size_t idx = 0;

  auto LIH = labelInputs.getHandle<size_t>();
  auto IIH = imageInputs.getHandle<>();

  for (unsigned w = 0; w < mnistNumImages; w++) {
    LIH.at({w, 0}) = labels[w];
    for (unsigned x = 0; x < 28; x++) {
      for (unsigned y = 0; y < 28; y++) {
        IIH.at({w, x, y, 0}) = imagesAsFloatPtr[idx++];
      }
    }
  }
  size_t numImages = labels.size();
  GLOW_ASSERT(numImages && "No images were found.");
  return numImages;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "TFLite execution test\n\n");

  llvm::outs() << "Loading the mnist database.\n";

  Tensor imageInputs;
  Tensor labelInputs;

  unsigned numImages = loadMNIST(imageInputs, labelInputs);
  llvm::outs() << "Loaded " << numImages << " images.\n";

  ExecutionEngine EE(executionBackend);
  auto &mod = EE.getModule();

  auto F = mod.createFunction("main");
  TFLiteLoader LD(modelFile.c_str(), {}, {}, *F);

  // Make sure that graph can be compiled and run.
  EE.compile(CompilationMode::Infer, F);

  auto LIH = labelInputs.getHandle<size_t>();

  // Check how many examples out of eighty previously unseen digits we can
  // classify correctly.
  int rightAnswer = 0;

  auto A = llvm::cast<Variable>(LD.getNodeByName("input"));
  assert(A);
  auto result = LD.getRoot();
  assert(result);

  for (int iter = 0; iter < 100; iter++) {
    Tensor sample(ElemKind::FloatTy, A->dims());
    sample.copyConsecutiveSlices(&imageInputs, iter);
    llvm::outs() << "MNIST Input";
    auto I = sample.getHandle<>().extractSlice(0);
    I.getHandle<>().dumpAscii();

    // quantize input data
    vector<int8_t> input_data;
    for (auto i = 0; i < sample.size(); ++i) {
      auto data = sample.getRawDataPointer<float>();
      input_data.push_back(quantization::quantize(data[i], {1 / 256., -128}));
    }

    Tensor input((void *)&input_data[0], A->getType());
    EE.run({A}, {&input});

    Tensor &res = result->getVariable()->getPayload();

    size_t guess = res.getHandle<int8_t>().minMaxArg().second;

    size_t correct = LIH.at({(size_t)iter, 0});
    rightAnswer += (guess == correct);

    llvm::outs() << "Expected: " << correct << " Guessed: " << guess << "\n";

    res.getHandle<int8_t>().dump();
    llvm::outs() << "\n-------------\n";
  }

  llvm::outs() << "Results: guessed/total:" << rightAnswer << "/" << 100
               << "\n";
  GLOW_ASSERT(rightAnswer > 96 &&
              "Did not classify as many digits as expected");
  return 0;
}
