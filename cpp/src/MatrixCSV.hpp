#pragma once
#include "IMatrix.hpp"
#include <string>
#include <vector>


// load matrix from CSV, provide IMatrix interface
class MatrixCSV : public IMatrix
{
public:
	explicit MatrixCSV(const std::string& filename);

	virtual double getAt(size_t row, size_t col) const;
	virtual void setAt(size_t row, size_t col, double val);
	size_t rows() const { return m_rows; }
	size_t cols() const { return m_cols; }

private:
	std::vector<std::vector<double>> m_data;
};

