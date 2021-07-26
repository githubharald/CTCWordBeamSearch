#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <cctype>
#include <memory>
#include <exception>
#include <cstddef>
#include <stdint.h>
#include "MatrixArray.hpp"
#include "WordBeamSearch.hpp"
#include "LanguageModel.hpp"


namespace py = pybind11;


// pybind11 NumPy interface
class NPWordBeamSearch
{
private:
	std::shared_ptr<LanguageModel> m_lm;
	size_t m_beamWidth = 0;
	size_t m_numChars = 0;
	LanguageModelType m_lmType = LanguageModelType::Words;

public:
	// CTOR
	NPWordBeamSearch(size_t beamWidth, std::string lmType, float lmSmoothing, const std::string& corpus, const std::string& chars, const std::string& wordChars)
	{
		m_beamWidth = beamWidth;

		// map string to enum
		std::transform(lmType.begin(), lmType.end(), lmType.begin(), tolower);
		if (lmType == "words")
		{
			m_lmType = LanguageModelType::Words;
		}
		else if (lmType == "ngrams")
		{
			m_lmType = LanguageModelType::NGrams;
		}
		else if (lmType == "ngramsforecast")
		{
			m_lmType = LanguageModelType::NGramsForecast;
		}
		else if (lmType == "ngramsforecastandsample")
		{
			m_lmType = LanguageModelType::NGramsForecastAndSample;
		}
		else
		{
			throw std::invalid_argument("unknown LM type (lmType)");
		}

		// create language model
		m_lm = std::make_shared<LanguageModel>(corpus, chars, wordChars, m_lmType, lmSmoothing);

		// query number of chars (may be different to chars.size()) to check tensor shape
		m_numChars = m_lm->getAllChars().size();

		// check string sizes now and the mat size later in the compute method
		const size_t numWordChars = m_lm->getWordChars().size();
		if (!(numWordChars > 0 && numWordChars <= m_numChars))
		{
			throw std::invalid_argument("check length of chars and wordChars: 0<len(wordChars)<=len(chars)");
		}

	}


	// method which gets a NumPy array (TxBxC) as input and returns a list of B lists (label-strings)
	std::vector<std::vector<uint32_t>> compute(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const
	{
		py::buffer_info buf = array.request();
		const size_t maxT = buf.shape[0];
		const size_t maxB = buf.shape[1];
		const size_t maxC = buf.shape[2];

		// check tensor size
		if (maxC != m_numChars + 1)
		{
			throw std::invalid_argument("the number of characters (chars) plus 1  must equal dimension 2 of the input tensor (mat)");
		}

		// go over all batch elements
		std::vector<std::vector<uint32_t>> res;
		for (size_t b = 0; b < maxB; ++b)
		{
			// wrapper around Tensor
			MatrixArray mat(array, b, maxT, maxC);

			// apply decoding algorithm to batch element 
			res.push_back(wordBeamSearch(mat, m_beamWidth, m_lm, m_lmType));
		}

		return res;
	}
};


// register C++ class "NPWordBeamSearch" as "WordBeamSearch" in Python
PYBIND11_MODULE(word_beam_search, m) {
	py::class_<NPWordBeamSearch>(m, "WordBeamSearch")
		.def(py::init<size_t, std::string, float, const std::string&, const std::string&, const std::string&>())
		.def("compute", &NPWordBeamSearch::compute);
}

