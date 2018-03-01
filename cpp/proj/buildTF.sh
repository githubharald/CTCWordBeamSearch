#!/bin/bash

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -Wall -O2 --std=c++14 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC 
