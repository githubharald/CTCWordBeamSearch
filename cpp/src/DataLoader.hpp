#pragma once
#include "MatrixCSV.hpp"
#include "LanguageModel.hpp"
#include <string>
#include <vector>
#include <memory>
#include <stdint.h>
#include <cstddef>


// load sample data, create LM from it, iterate over samples
class DataLoader
{
public:
	// sample with matrix to be decoded and ground truth text
	struct Data
	{
		MatrixCSV mat;
		std::vector<uint32_t> gt;
	};

	// CTOR. Path points to directory holding files corpus.txt, chars.txt, wordChars.txt and samples mat_X.csv and gt_X.txt with X in {0, 1, 2, ...}
	DataLoader(const std::string& path, size_t sampleEach, LanguageModelType lmType, double addK=0.0);

	// get LM
	std::shared_ptr<LanguageModel> getLanguageModel() const;

	// iterator interface
	Data getNext() const;
	bool hasNext() const;

private:
	std::string m_path;
	std::shared_ptr<LanguageModel> m_lm;
	mutable size_t m_currIdx=0;
	const size_t m_sampleEach = 1;

	void applySoftmax(MatrixCSV& mat) const;
	bool fileExists(const std::string& path) const;
};


