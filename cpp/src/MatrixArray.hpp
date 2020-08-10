#pragma once
#include "IMatrix.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;


class MatrixArray : public IMatrix
{
public:
	MatrixArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array, size_t b, size_t maxT, size_t maxC)
	:m_array(array)
	,m_batch(b)
	{
		m_rows=maxT;
		m_cols=maxC;
	}

	virtual double getAt(size_t row, size_t col) const
	{	
		return m_array.at(row, m_batch, col);
	}
	
	virtual void setAt(size_t row, size_t col, double val)
	{
		// not implemented
	}

private:
	const py::array_t<double, py::array::c_style | py::array::forcecast>& m_array;
	size_t m_batch;
};


