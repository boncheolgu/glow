#include "glow/Importer/TFLite.h"

#include <cassert>
#include <fstream>
#include <vector>

#include "glow/Quantization/Quantization.h"

#include "llvm/Support/raw_ostream.h"

#include "tflite_generated.h"

using namespace std;
using namespace glow;

namespace {
static glow::Type tensor_type(const tflite::Tensor *tensor) {
  using namespace tflite;

  const auto kind = [&] {
    switch (tensor->type()) {
    case TensorType_UINT8:
      return ElemKind::Int8QTy;
    case TensorType_INT32:
      return ElemKind::Int32QTy;
    default:
      GLOW_ASSERT(false && "unsupported type.");
    }
  }();

  GLOW_ASSERT(tensor->shape() && tensor->shape()->Length());
  const vector<size_t> dims(tensor->shape()->begin(), tensor->shape()->end());

  if (auto quantization = tensor->quantization()) {
    GLOW_ASSERT(quantization->scale() && quantization->scale()->Length() == 1);
    GLOW_ASSERT(quantization->zero_point() &&
                quantization->zero_point()->Length() == 1);

    // TFLite has an element type of `uint8_t`, thus convert offset values to
    // `int8_t` by substracting 128.
    return Type{kind, dims, quantization->scale()->Get(0),
                static_cast<int32_t>(quantization->zero_point()->Get(0) - 128)};
  }

  return Type{kind, dims};
}

// https://www.tensorflow.org/api_guides/python/nn#Convolution
std::pair<size_t, size_t> calculate_same_padding(size_t input, size_t filter,
                                                 size_t stride) {
  auto remainder = input % stride;
  const auto pad_along = (remainder == 0)
                             ? (filter > stride ? filter - stride : 0)
                             : (filter > remainder ? filter - remainder : 0);

  return {pad_along / 2, pad_along - pad_along / 2};
}

struct Conv2DParams {
  size_t kernel, stride, padding[4], group;
};

Conv2DParams getConv2DParams(const Node *input, const Node *filter,
                             const tflite::Conv2DOptions *options) {
  GLOW_ASSERT(options->stride_w() == options->stride_h());

  const auto filter_shape = ShapeNHWC{filter->dims(0)};
  GLOW_ASSERT(filter_shape.h == filter_shape.w);

  const auto input_shape = ShapeNHWC{input->dims(0)};

  switch (options->padding()) {
  case tflite::Padding_SAME: {
    const auto pad_h = calculate_same_padding(input_shape.h, filter_shape.h,
                                              options->stride_h());
    const auto pad_w = calculate_same_padding(input_shape.w, filter_shape.w,
                                              options->stride_w());

    return Conv2DParams{filter_shape.h,
                        static_cast<size_t>(options->stride_h()),
                        // right-upper aligned filter
                        {pad_h.first, pad_w.first, pad_h.second, pad_w.second},
                        1};
  }
  case tflite::Padding_VALID: {

    return Conv2DParams{filter_shape.h,
                        static_cast<size_t>(options->stride_h()),
                        // right-upper aligned filter
                        {0, 0, 0, 0},
                        1};
  }
  default:
    GLOW_ASSERT(false && "invalid padding");
  }
}

Conv2DParams getConv2DParams(const Node *input, const Node *filter,
                             const tflite::DepthwiseConv2DOptions *options) {
  GLOW_ASSERT(options->stride_w() == options->stride_h());

  const auto filter_shape = ShapeNHWC{filter->dims(0)};
  GLOW_ASSERT(filter_shape.h == filter_shape.w);

  const auto input_shape = ShapeNHWC{input->dims(0)};

  switch (options->padding()) {
  case tflite::Padding_SAME: {
    const auto pad_h = calculate_same_padding(input_shape.h, filter_shape.h,
                                              options->stride_h());
    const auto pad_w = calculate_same_padding(input_shape.w, filter_shape.w,
                                              options->stride_w());

    return Conv2DParams{filter_shape.h,
                        static_cast<size_t>(options->stride_h()),
                        // right-upper aligned filter
                        {pad_h.first, pad_w.first, pad_h.second, pad_w.second},
                        input_shape.c};
  }
  case tflite::Padding_VALID: {

    return Conv2DParams{filter_shape.h,
                        static_cast<size_t>(options->stride_h()),
                        // right-upper aligned filter
                        {0, 0, 0, 0},
                        input_shape.c};
  }
  default:
    GLOW_ASSERT(false && "invalid padding");
  }
}

Tensor convertToInt8Bias(const llvm::ArrayRef<int32_t> &data,
                         const Type &type) {
  const int32_t *q_min, *q_max;
  tie(q_min, q_max) = minmax_element(data.begin(), data.end());
  double q_scale = type.getScale();

  // calculate new scale and offset
  auto b_scale = (*q_max - *q_min) * q_scale / 255;
  auto b_offset = round(-1 * *q_min * q_scale / b_scale) - 128;

  // create tensor containing elements of int8_t from `data`
  glow::Tensor tensor{ElemKind::Int8QTy, type.dims(),
                      static_cast<float>(b_scale),
                      static_cast<int32_t>(b_offset)};
  std::transform(data.begin(), data.end(), tensor.getRawDataPointer<int8_t>(),
                 [&](auto n) {
                   return quantization::clip<int32_t, int8_t>(
                       round(n * q_scale / b_scale + b_offset));
                 });
  return tensor;
}
} // namespace

TFLiteLoader::TFLiteLoader(const char *filename,
                           llvm::ArrayRef<const char *> names,
                           llvm::ArrayRef<Tensor *> tensors, Function &F)
    : ProtobufLoader(names, tensors, F) {
  ifstream fs(filename, std::ios::binary);
  GLOW_ASSERT(fs.is_open());

  std::vector<char> flat_buffer{std::istreambuf_iterator<char>(fs),
                                std::istreambuf_iterator<char>()};

  rootModel_ = tflite::GetModel(flat_buffer.data());
  llvm::outs() << "Loading the " << filename << ": \""
               << rootModel_->description()->c_str() << "\"\n";

  auto subgraphs = rootModel_->subgraphs();
  for (auto sg_id = 0; sg_id < subgraphs->Length(); ++sg_id) {
    auto subgraph = subgraphs->Get(sg_id);

    build_function(subgraph);
  }
}

const std::vector<Variable *> &TFLiteLoader::getInputs() const {
  return inputs_;
}

const std::vector<SaveNode *> &TFLiteLoader::getOutputs() const {
  return outputs_;
}

void TFLiteLoader::build_function(const tflite::SubGraph *subgraph) {
  using namespace tflite;

  // build a table mapping an output tensor to an operator which
  // produces the tensor as an output.
  operator_table out2op(subgraph->tensors()->Length());
  auto operators = subgraph->operators();
  for (auto op_id = 0; op_id < operators->Length(); ++op_id) {
    auto op = operators->Get(op_id);
    for (auto out : *op->outputs()) {
      GLOW_ASSERT(out < out2op.size());

      out2op[out] = operator_index{op_id};
    }
  }

  for (auto out_id = 0; out_id < subgraph->outputs()->Length(); ++out_id) {
    auto output = subgraph->outputs()->Get(out_id);

    // FIXME: Add 128 to convert int8_t to uint8_t
    auto producer = create_node(subgraph, out2op, tensor_index{output});
    outputs_.push_back(G_.createSave("result" + to_string(out_id), producer));
  }

  if (!outputs_.empty()) {
    root_ = outputs_[0];
  }

  // FIXME: Subtract 128 to convert uint8_t to int8_t
  // for (const auto input : inputs) {
  // }
}

Node *TFLiteLoader::create_node(const tflite::SubGraph *subgraph,
                                operator_table &out2op,
                                tensor_index tensor_id) {
  using namespace tflite;

  const auto tensors = subgraph->tensors();
  const auto operators = subgraph->operators();

  const auto result_tensor = tensors->Get(static_cast<int32_t>(tensor_id));
  const auto result_type = tensor_type(result_tensor);

  if (auto op_id = out2op[static_cast<int32_t>(tensor_id)]) {
    auto op = operators->Get(static_cast<int32_t>(*op_id));
    auto opcode = rootModel_->operator_codes()->Get(op->opcode_index());

    auto builtin_code = opcode->builtin_code();
    switch (builtin_code) {
    case BuiltinOperator_CONV_2D: {
      GLOW_ASSERT(op->inputs()->Length() == 3);
      const auto input =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(0)});
      const auto filter =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(1)});
      const auto bias =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(2)});

      const auto options = op->builtin_options_as_Conv2DOptions();
      GLOW_ASSERT(options);

      const auto conv_param = getConv2DParams(input, filter, options);

      const auto node = G_.createConv(
          "conv" + to_string(static_cast<int32_t>(*op_id)), input, filter, bias,
          &result_type, conv_param.kernel, conv_param.stride,
          conv_param.padding, conv_param.group);

      nodeByName_.emplace(result_tensor->name()->c_str(), node);
      return node;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      GLOW_ASSERT(op->inputs()->Length() == 3);
      const auto input =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(0)});

      // Transpose the filter tensor cause TFLite has a different layout from
      // that of glow when using group parameter.
      const auto filter = G_.createTranspose(
          "transpose" + to_string(static_cast<int32_t>(*op_id)),
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(1)}),
          {3, 1, 2, 0});

      const auto bias =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(2)});

      const auto options = op->builtin_options_as_DepthwiseConv2DOptions();
      GLOW_ASSERT(options);

      const auto conv_param = getConv2DParams(input, filter, options);

      const auto node = G_.createConv(
          "dconv" + to_string(static_cast<int32_t>(*op_id)), input, filter,
          bias, &result_type, conv_param.kernel, conv_param.stride,
          conv_param.padding, conv_param.group);

      nodeByName_.emplace(result_tensor->name()->c_str(), node);
      return node;
    }
    case BuiltinOperator_AVERAGE_POOL_2D: {
      GLOW_ASSERT(op->inputs()->Length() == 1);
      const auto input =
          create_node(subgraph, out2op, tensor_index{op->inputs()->Get(0)});

      const auto options = op->builtin_options_as_Pool2DOptions();
      GLOW_ASSERT(options);

      GLOW_ASSERT(options->filter_width() == options->filter_height());
      const auto kernel = options->filter_height();

      GLOW_ASSERT(options->stride_w() == options->stride_h());
      const auto stride = options->stride_w();

      const auto padding = [&] {
        switch (options->padding()) {
        case Padding_SAME: {
          return static_cast<size_t>(kernel / 2);
        }
        case Padding_VALID: {
          return static_cast<size_t>(0);
        }
        default:
          GLOW_ASSERT(false && "invalid padding");
        }
      }();

      const auto node =
          G_.createPoolAvg("avgpool" + to_string(static_cast<int32_t>(*op_id)),
                           input, kernel, stride, padding);

      nodeByName_.emplace(result_tensor->name()->c_str(), node);
      return node;
    }
    case BuiltinOperator_RESHAPE:
    case BuiltinOperator_SQUEEZE:
    case BuiltinOperator_SOFTMAX: {
      // GLOW_ASSERT(op->inputs()->Length() == 1);
      llvm::errs() << EnumNameBuiltinOperator(builtin_code) << " ignored.\n";
      return create_node(subgraph, out2op, tensor_index{op->inputs()->Get(0)});
    }
    default: { GLOW_ASSERT(false && "unexpeced operator given."); }
    }
  }

  // Here, `result_tensor` represents an input variable.
  const auto buffer = rootModel_->buffers()->Get(result_tensor->buffer());
  if (auto data = buffer->data()) {
    // Here, the tensor is a prepared parameter such as filter and bias.
    GLOW_ASSERT(data->end() - data->begin() == data->Length());

    Tensor tensor;
    switch (result_tensor->type()) {
    case TensorType_UINT8: { // Here, this tensor is weight of uint8_t.
      GLOW_ASSERT(result_type.size() == data->Length());

      tensor.reset(result_type);
      transform(data->begin(), data->end(), tensor.getRawDataPointer<int8_t>(),
                [](auto n) {
                  // TFLite has an element type of `uint8_t`, thus convert
                  // elements to `int8_t` by substracting 128.
                  return n - 128;
                });
      break;
    }
    case TensorType_INT32: { // Here, this tensor is bias of int32_t
      GLOW_ASSERT(result_type.size() * sizeof(int32_t) == data->Length());

      // offset of TFLite bias should be 0 of uint8_t, which is -128 of
      // int8_t.
      GLOW_ASSERT(result_type.getOffset() == -128);

      const llvm::ArrayRef<int32_t> int32_data{
          reinterpret_cast<const int32_t *>(data->Data()), result_type.size()};

      tensor = convertToInt8Bias(int32_data, result_type);
      break;
    }
    default:
      GLOW_ASSERT(false && "unsupported type.");
      break;
    }

    auto param_var = G_.getParent()->createVariable(
        tensor.getElementType(), tensor.dims(), tensor.getType().getScale(),
        tensor.getType().getOffset(),
        "[" + to_string(static_cast<int32_t>(tensor_id)) + "]" +
            result_tensor->name()->str(),
        VisibilityKind::Private, Variable::TrainKind::None);

    param_var->getPayload() = std::move(tensor);

    nodeByName_.emplace(result_tensor->name()->c_str(), param_var);
    return param_var;
  }

  // Here, the tensor is an input, which will be given at the execution time.
  const auto input_var = G_.getParent()->createVariable(
      result_type.getElementType(), result_type.dims(), result_type.getScale(),
      result_type.getOffset(), result_tensor->name()->str(),
      VisibilityKind::Public, Variable::TrainKind::None);

  inputs_.push_back(input_var);
  nodeByName_.emplace(result_tensor->name()->c_str(), input_var);
  return input_var;
}
