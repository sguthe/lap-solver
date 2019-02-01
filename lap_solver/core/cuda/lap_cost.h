#pragma once
#include <iostream>

namespace lap
{
	namespace cuda
	{
		// RowCostFunction is the only interface supported for CUDA at the moment
		template <class TC, typename GETCOSTROW>
		class RowCostFunction
		{
		protected:
			GETCOSTROW getcostrow;
			TC initialEpsilon;
		public:
			RowCostFunction(GETCOSTROW &getcostrow) : getcostrow(getcostrow), initialEpsilon(0) {}
			~RowCostFunction() {}
		public:
			__forceinline const TC getInitialEpsilon() const { return initialEpsilon; }
			__forceinline void setInitialEpsilon(TC eps) { initialEpsilon = eps; }
			__forceinline void getCostRow(TC *row, int t, cudaStream_t stream, int x, int start, int end) const { getcostrow(row, t, stream, x, start, end); }
		};
	}
}
