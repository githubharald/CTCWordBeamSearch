#include "MatrixCSV.hpp"
#include <fstream>


MatrixCSV::MatrixCSV(const std::string& filename)
{
	std::ifstream f(filename);
	std::string line;
	
	while (std::getline(f, line))
	{
		std::vector<double> row;
		std::string tmp;
		for (size_t i = 0; i < line.size(); ++i)
		{
			char c = line[i];
			if (c == ';' || i + 1 == line.size())
			{
				row.push_back(std::stod(tmp));
				tmp.clear();
			}
			else
			{
				tmp.push_back(c);
			}
		}

		if (!row.empty())
		{
			m_data.push_back(row);
		}
	}

	m_rows = m_data.size();
	m_cols = m_data.front().size();
}


double MatrixCSV::getAt(size_t row, size_t col) const
{
	return m_data[row][col];
}


void MatrixCSV::setAt(size_t row, size_t col, double val)
{
	m_data[row][col] = val;
}


