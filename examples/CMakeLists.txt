
add_executable(cifar10
                 cifar10.cpp)
target_link_libraries(cifar10
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support)

add_executable(mnist
                 mnist.cpp)
target_link_libraries(mnist
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support)

add_executable(ptb
                 ptb.cpp)
target_link_libraries(ptb
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support)

add_executable(char-rnn
                 char-rnn.cpp)
target_link_libraries(char-rnn
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        IR
                        Support)

add_executable(fr2en
                 fr2en.cpp)
target_link_libraries(fr2en
                      PRIVATE
                        Base
                        ExecutionEngine
                        IR
                        Quantization
                        Support)

add_executable(tflite tflite.cpp)
target_link_libraries(tflite
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        Importer
                        IR
                        Support)
