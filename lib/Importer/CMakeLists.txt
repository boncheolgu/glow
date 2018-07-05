find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS caffe.proto onnx.proto)

add_library(Importer
            ProtobufLoader.cpp
            Caffe2.cpp
            ONNX.cpp
            TFLite.cpp
            ${PROTO_SRCS}
            ${PROTO_HDRS})

# this causes build error for GCC
# set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS --gen-nullable)
BUILD_FLATBUFFERS(tflite.fbs
  "${CMAKE_CURRENT_SOURCE_DIR}"
  tflite_schema
  flatc
  "${CMAKE_CURRENT_BINARY_DIR}"
  ""
  "")

add_dependencies(Importer tflite_schema)

target_compile_definitions(Importer
                           INTERFACE
                           -DGOOGLE_PROTOBUF_NO_RTTI)

target_include_directories(Importer PRIVATE ${FLATBUFFERS_INCLUDE_DIR})
target_link_libraries(Importer
                      PRIVATE
                        Base
                        Graph
                        Support
                        ${PROTOBUF_LIBRARY})
