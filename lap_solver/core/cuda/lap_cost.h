#pragma once
#include <iostream>

namespace lap
{
	namespace cuda
	{
		// RowCostFunction is the only interface supported for CUDA at the moment
		template <class TC, typename GETCOSTROW, typename GETCOST>
		class RowCostFunction
		{
		protected:
			GETCOSTROW getcostrow;
			GETCOST getcost;
			TC initialEpsilon;
		public:
			RowCostFunction(GETCOSTROW &getcostrow, GETCOST &getcost) : getcostrow(getcostrow), getcost(getcost), initialEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end) const { getcostrow(row, t, stream, x, start, end); }
			__forceinline void getCost(TC *row, cudaStream_t stream, int *rowsol, int dim2) const { getcost(row, stream, rowsol, dim2); }
		};
	}
}
