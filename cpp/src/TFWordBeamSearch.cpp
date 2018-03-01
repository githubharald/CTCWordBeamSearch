#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <algorithm>
#include <string>
#include <cctype>
#include <memory>
#include <exception>
#include <cstddef>
#include <stdint.h>
#include "MatrixTensor.hpp"
#include "WordBeamSearch.hpp"
#include "LanguageModel.hpp"


REGISTER_OP("WordBeamSearch")
.Input("mat: float32")
.Attr("beamWidth: int")
.Attr("lmType: string")
.Attr("lmSmoothing: float")
.Attr("corpus: string")
.Attr("chars: string")
.Attr("wordChars: string")
.Output("result: int32")
.Doc(
"Decodes matrix (mat) using a dictionary and language model created from text corpus (corpus). "\
"The softmax function must be applied in advance to mat. "\
"All characters (chars) must be passed in the same order as they appear in mat, not including the CTC-blank. "\
"The characters (wordChars) which can occur in a word are used to create the dictionary and language model from the corpus. "\
"The LM scoring mode (lmType) must be one of the following four strings (not case-sensitive): 'Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample'. "\
"Pass strings UTF8 encoded if using special characters. "
);


using namespace tensorflow;


// custom TF op
class TFWordBeamSearch : public OpKernel 
{
private:
	std::shared_ptr<LanguageModel> m_lm;
	size_t m_beamWidth=0;
	size_t m_numChars=0;
	LanguageModelType m_lmType=LanguageModelType::Words;

public:
	// CTOR
	explicit TFWordBeamSearch(OpKernelConstruction* context) 
	:OpKernel(context) 
	{
		// read beam width
		int64 beamWidth64=0;
		OP_REQUIRES_OK(context, context->GetAttr("beamWidth", &beamWidth64));
		m_beamWidth=static_cast<size_t>(beamWidth64);
		
		// read type of language model
		std::string strLmType;
		OP_REQUIRES_OK(context, context->GetAttr("lmType", &strLmType));
		
		// map string to enum (default is Words)
		std::transform(strLmType.begin(), strLmType.end(), strLmType.begin(), tolower);
		if(strLmType=="words")
		{
			m_lmType=LanguageModelType::Words;
		}
		else if(strLmType=="ngrams")
		{
			m_lmType=LanguageModelType::NGrams;
		}
		else if(strLmType=="ngramsforecast")
		{
			m_lmType=LanguageModelType::NGramsForecast;
		}
		else if(strLmType=="ngramsforecastandsample")
		{
			m_lmType=LanguageModelType::NGramsForecastAndSample;
		}
		else
		{
			throw std::invalid_argument("unknown LM type (lmType)");
		}

		// read lm smoothing value	
		float lmSmoothing=0.0;
		OP_REQUIRES_OK(context, context->GetAttr("lmSmoothing", &lmSmoothing));

		// read corpus (utf8)
		std::string corpus;
		OP_REQUIRES_OK(context, context->GetAttr("corpus", &corpus));

		// read all chars (utf8)
		std::string chars;
		OP_REQUIRES_OK(context, context->GetAttr("chars", &chars));

		// read all chars which occur inside words
		std::string wordChars;
		OP_REQUIRES_OK(context, context->GetAttr("wordChars", &wordChars));

		// create language model
		m_lm=std::make_shared<LanguageModel>(corpus, chars, wordChars, m_lmType, lmSmoothing);

		// query number of chars (may be different to chars.size()) to check tensor shape
		m_numChars=m_lm->getAllChars().size();

		// check string sizes now and the mat size later in the Compute method
		const size_t numWordChars=m_lm->getWordChars().size();
		if(!(numWordChars>0 && numWordChars<m_numChars))
		{
			throw std::invalid_argument("wordChars must contain at least one character and at least one character less than chars: 0<len(wordChars)<len(chars)");
		}

	}


	// computation in TF graph
	void Compute(OpKernelContext* context) override 
	{
		// input: TxBxC, float32
		const Tensor& inputTensor = context->input(0);
		const auto inputShape=inputTensor.shape();
		const auto T=inputShape.dim_size(0);
		const auto B=inputShape.dim_size(1);
		const auto C=inputShape.dim_size(2);
		const size_t blank=C-1;

		// check tensor size
		if(static_cast<size_t>(C)!=m_numChars+1)
		{
			throw std::invalid_argument("the number of characters (chars) plus 1  must equal dimension 2 of the tensor (mat)");
		}

		// input tensor
		auto inputMapped=inputTensor.tensor<float,3>();

		// output: BxT, int32
		Tensor* outputTensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({B, T}), &outputTensor));
		auto outputMapped=outputTensor->tensor<int32,2>();

		// go over all batch elements
		for(int b=0; b<B; ++b)
		{
			// wrapper around Tensor
			MatrixTensor<decltype(inputMapped)> mat(inputMapped, b, T, C);

			// apply decoding algorithm to batch element 
			const std::vector<uint32_t> decoded=wordBeamSearch(mat, m_beamWidth, m_lm, m_lmType);

			// write to output tensor
			for(int t=0; t<T; ++t)
			{
				outputMapped(b, t)=t<static_cast<int>(decoded.size()) ? decoded[t] : blank;
			}
		}
	}
};


REGISTER_KERNEL_BUILDER(Name("WordBeamSearch").Device(DEVICE_CPU), TFWordBeamSearch);

