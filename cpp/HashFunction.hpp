#pragma once
#include <vector>
#include <stdint.h>
#include <cstddef>


// calculate hash for a beam text
struct HashFunction
{
	size_t operator()(const std::vector<uint32_t>& text) const
	{
		// taken from: https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		std::size_t res = text.size();
		for (const auto& l : text)
		{
			res ^= l + 0x9e3779b9 + (res << 6) + (res >> 2);
		}
		return res;
	}
};
