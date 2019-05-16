cmake_minimum_required(VERSION 2.8)

project(WordBeamSearch)

set (CMAKE_CXX_STANDARD 11)

add_executable(
    WordBeamSearch.out
    main.cpp WordBeamSearch.cpp PrefixTree.cpp Metrics.cpp MatrixCSV.cpp LanguageModel.cpp DataLoader.cpp Beam.cpp
)