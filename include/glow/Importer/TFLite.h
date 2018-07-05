#ifndef GLOW_IMPORTER_TFLITE_H
#define GLOW_IMPORTER_TFLITE_H

#include <cstdint>
#include <vector>

#include "glow/Importer/ProtobufLoader.h"

#include "llvm/ADT/Optional.h"

namespace tflite {
struct SubGraph;
struct Model;
} // namespace tflite

namespace glow {

class Node;
class Variable;
class Function;
class SaveNode;
class Module;

using tensor_index = int32_t;
using operator_index = int32_t;

struct TFLiteFunction {
  std::vector<Variable *> inputs;
  std::vector<SaveNode *> outputs;
};

class TFLiteLoader : public ProtobufLoader {
public:
  TFLiteLoader(const char *filename, llvm::ArrayRef<const char *> names,
               llvm::ArrayRef<Tensor *> tensors, Function &F);

  const std::vector<Variable *> &getInputs() const;
  const std::vector<SaveNode *> &getOutputs() const;

private:
  const tflite::Model *rootModel_;
  std::vector<Variable *> inputs_;
  std::vector<SaveNode *> outputs_;

  using operator_table = std::vector<llvm::Optional<operator_index>>;

  void build_function(const tflite::SubGraph *subgraph);

  Node *create_node(const tflite::SubGraph *subgraph, operator_table &out2op,
                    tensor_index tensor_id);
};

} // namespace glow
#endif // GLOW_IMPORTER_TFLITE_H
