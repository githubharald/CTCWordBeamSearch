#!/bin/bash

g++ -Wall -O2 --std=c++11 -o WordBeamSearch.out ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp
