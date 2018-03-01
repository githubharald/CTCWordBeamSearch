#include "DataLoader.hpp"
#include <fstream>
#include <streambuf>
#include <math.h>


DataLoader::DataLoader(const std::string& path, size_t sampleEach, LanguageModelType lmType, double addK)
:m_path(path)
,m_sampleEach(sampleEach)
{
	// open text files
	std::ifstream corpusFile{m_path+"/corpus.txt"};
	std::ifstream charsFile{m_path+"/chars.txt"};
	std::ifstream wordCharsFile{m_path+"/wordChars.txt"};

	// read text files
	std::string corpus{ std::istreambuf_iterator<char>(corpusFile), std::istreambuf_iterator<char>() };
	std::string chars{ std::istreambuf_iterator<char>(charsFile), std::istreambuf_iterator<char>() };
	std::string wordChars{ std::istreambuf_iterator<char>(wordCharsFile), std::istreambuf_iterator<char>() };

	// create language model
	m_lm = std::make_shared<LanguageModel>(corpus, chars, wordChars, lmType, addK);
}


std::shared_ptr<LanguageModel> DataLoader::getLanguageModel() const
{
	return m_lm;
}


DataLoader::Data DataLoader::getNext() const
{
	// filename
	const std::string matFilename = m_path + "/mat_" + std::to_string(m_currIdx) + ".csv";
	const std::string gtFilename = m_path + "/gt_" + std::to_string(m_currIdx) + ".txt";

	
	// read csv and return result
	MatrixCSV mat(matFilename);
	applySoftmax(mat);
	std::ifstream ftFile(gtFilename);
	std::string gt{ std::istreambuf_iterator<char>{ftFile}, std::istreambuf_iterator<char>() };
	Data res{ mat, m_lm->utf8ToLabel(gt) };
	m_currIdx += m_sampleEach;
	return res;
}


bool DataLoader::hasNext() const
{
	// check if matrix and ground truth with given index exist
	const std::string matFilename = m_path + "/mat_" + std::to_string(m_currIdx) + ".csv";
	const std::string gtFilename = m_path + "/gt_" + std::to_string(m_currIdx) + ".txt";

	return fileExists(matFilename) && fileExists(gtFilename);
}


void DataLoader::applySoftmax(MatrixCSV& mat) const
{
	for (size_t t = 0; t < mat.rows(); ++t)
	{
		// sum up
		double sum = 0.0;
		for (size_t c = 0; c < mat.cols(); ++c)
		{
			sum += exp(mat.getAt(t, c));
		}

		// normalize prob distribution
		for (size_t c = 0; c < mat.cols(); ++c)
		{
			mat.setAt(t, c, exp(mat.getAt(t, c)) / sum);
		}
	}
}


bool DataLoader::fileExists(const std::string& path) const
{
	std::ifstream f(path);
	return f.good();
}

