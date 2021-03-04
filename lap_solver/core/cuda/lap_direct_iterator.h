#pragma once

#include "lap_worksharing.h"
#include <variant>

namespace lap
{
	namespace cuda
	{
		template <class TC, class CF>
		class DirectIterator
		{
		public:
			CF& costfunc;
			Worksharing& ws;
			static const bool GPU = false;

		public:
			DirectIterator(CF& costfunc, Worksharing& ws) : costfunc(costfunc), ws(ws) {}
			~DirectIterator() {}

			void getHitMiss(long long& hit, long long& miss) { hit = miss = 0; }

			__forceinline const TC* getRow(int t, int i, bool async) { return costfunc.getRow(t, i); }
		};

		// empty state
		typedef std::monostate DeviceDirectIteratorState;

		template <class TC, class CF>
		class DeviceDirectIteratorObject
		{
		protected:
			CF& costfunc;
		public:
			DeviceDirectIteratorObject(CF& costfunc) : costfunc(costfunc) { }

			~DeviceDirectIteratorObject() { }

			template <class ISTATE, class STATE>
			__forceinline __device__ void openRowWarp(int i, int j, int start, ISTATE& istate, STATE& state, int& idx) { }

			template <class ISTATE, class STATE>
			__forceinline __device__ void openRowBlock(int i, int j, int start, ISTATE& istate, STATE& state, int& idx) { }

			template <class ISTATE>
			__forceinline __device__ void closeRow(ISTATE& istate) { }

			template <class ISTATE, class STATE>
			__forceinline __device__ TC getCost(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				return costfunc(i, j + start, state);
			}

			template <class ISTATE, class STATE>
			__forceinline __device__ TC getCostForced(int i, int j, int start, ISTATE& istate, STATE& state, int& idx)
			{
				return costfunc(i, j + start, state);
			}
		};

		template <class TC, class CF, class GETCOST>
		class DeviceDirectIterator
		{
		protected:
			DeviceDirectIteratorState istate;
			DeviceDirectIteratorObject<TC, GETCOST> iobject;
			CF& costfunc;
		public:
			Worksharing& ws;
			static const bool GPU = true;

		public:
			DeviceDirectIterator(CF& costfunc, GETCOST& getcost, Worksharing& ws) : iobject(getcost), costfunc(costfunc), ws(ws) {}
			~DeviceDirectIterator() {}

			void getHitMiss(long long& hit, long long& miss) { hit = miss = 0; }

			__forceinline decltype(costfunc.getState(0))& getState(int t) { return costfunc.getState(t); }

			__forceinline DeviceDirectIteratorState& getIState(int t) { return istate; }

			__forceinline DeviceDirectIteratorObject<TC, GETCOST>& getIObject() { return iobject; }
		};
	}
}
