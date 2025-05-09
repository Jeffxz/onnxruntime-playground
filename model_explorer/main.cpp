/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model should have float input
 *
 *
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <QCoreApplication>
#include <QDebug>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v)
{
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v)
{
  int total = 1;
  for (auto &i : v)
    total *= i;
  return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

int main(int argc, ORTCHAR_T *argv[])
{
  QCoreApplication app(argc, argv);
  if (argc < 2)
  {
    qDebug() << "Usage: " << argv[0] << "<onnx_model.onnx>";
    return -1;
  }

  std::basic_string<ORTCHAR_T> model_file = argv[1];

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  qDebug() << "Input Node Name/Shape (" << input_names.size() << "):";
  for (std::size_t i = 0; i < session.GetInputCount(); i++)
  {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    qDebug() << "\t" << input_names.at(i) << " : " << print_shape(input_shapes);
  }
  // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
  for (auto &s : input_shapes)
  {
    if (s < 0)
    {
      s = 1;
    }
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  qDebug() << "Output Node Name/Shape (" << output_names.size() << "):";
  for (std::size_t i = 0; i < session.GetOutputCount(); i++)
  {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    qDebug() << "\t" << output_names.at(i) << " : " << print_shape(output_shapes);
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes;
  auto total_number_elements = calculate_product(input_shape);

  // generate random numbers in the range [0, 255]
  std::vector<float> input_tensor_values(total_number_elements);
  std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&]
                { return rand() % 255; });
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
  qDebug() << "input_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape());

  // pass data through model
  std::vector<const char *> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  std::vector<const char *> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  qDebug() << "Running model...";
  try
  {
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());
    qDebug() << "Done!";

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
    assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
  }
  catch (const Ort::Exception &exception)
  {
    qDebug() << "ERROR running model inference: " << exception.what();
    exit(-1);
  }
}