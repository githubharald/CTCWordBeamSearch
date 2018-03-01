#include "WordBeamSearch.hpp"
#include "Beam.hpp"
#include <vector>
#include <memory>


std::vector<uint32_t> wordBeamSearch(const IMatrix& mat, size_t beamWidth, const std::shared_ptr<LanguageModel>& lm, LanguageModelType lmType)
{
	// dim0: T, dim1: C
	const size_t maxT = mat.rows();
	const size_t maxC = mat.cols();
	const size_t blank = maxC - 1;

	// initialise with genesis beam
	BeamList curr;
	BeamList last;
	const bool useNGrams = lmType == LanguageModelType::NGrams || lmType == LanguageModelType::NGramsForecast || lmType==LanguageModelType::NGramsForecastAndSample;
	const bool forcastNGrams = lmType == LanguageModelType::NGramsForecast || lmType == LanguageModelType::NGramsForecastAndSample;
	const bool sampleNGrams = lmType == LanguageModelType::NGramsForecastAndSample;
	last.addBeam(std::make_shared<Beam>(lm, useNGrams, forcastNGrams, sampleNGrams));

	// go over all time steps
	for (size_t t = 0; t < maxT; ++t)
	{
		// get k best beams and iterate 
		const std::vector<std::shared_ptr<Beam>> bestBeams = last.getBestBeams(beamWidth);
		for (const auto& beam : bestBeams)
		{
			double prBlank=0.0, prNonBlank=0.0;

			// calc prob that path ends with a non-blank
			prNonBlank = beam->getText().empty() ? 0.0 : beam->getNonBlankProb() * mat.getAt(t, beam->getText().back());

			// calc prob that path ends with a blank
			prBlank = beam->getTotalProb() * mat.getAt(t, blank);
			
			// add copy of original beam to current time step
			curr.addBeam(beam->createChildBeam(prBlank, prNonBlank));

			// extend current beam
			const std::vector<uint32_t> nextChars = beam->getNextChars();
			for (const auto c : nextChars)
			{
				prBlank = 0.0;
				prNonBlank = 0.0;
				// last char in beam equals new char: path must end with blank
				if (!beam->getText().empty() && beam->getText().back() == c)
				{
					prNonBlank = mat.getAt(t, c) * beam->getBlankProb();
				}
				// last char in beam and new char different
				else
				{
					prNonBlank = mat.getAt(t, c) * beam->getTotalProb();
				}

				curr.addBeam(beam->createChildBeam(prBlank, prNonBlank, c));
			}
		}

		last = std::move(curr);
	}

	// return best entry
	const auto bestBeam = last.getBestBeams(1)[0];
	bestBeam->completeText();
	return bestBeam->getText();
}


