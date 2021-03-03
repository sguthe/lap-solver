#pragma once
#include "../lap_cache.h"

namespace lap
{
	namespace sparse
	{
		// segmented least recently used
		class CacheSLRU : public lap::CacheSLRU
		{
		protected:
			std::vector<int> offset;
			std::vector<int> next;
			int open_idx;
			int open_off;
			int line_length;

		public:
			CacheSLRU() : lap::CacheSLRU() {
				open_idx = -1;
				open_off = -1;
			}
			~CacheSLRU() {}

			__forceinline void setSize(int entries, int dim, int length)
			{
				lap::CacheSLRU::setSize(entries, dim);
				line_length = length;
				offset.resize(dim);
				next.resize(dim);
				for (int i = 0; i < dim; i++) offset[i] = -1;
				for (int i = 0; i < dim; i++) next[i] = -1;
			}

			__forceinline bool find(int &idx, int &off, int i, int length)
			{
				if (map[i] == -1)
				{
					if ((open_idx >= 0) && (open_off + length <= line_length))
					{
						// append
						idx = open_idx;
						off = open_off;
						next[i] = id[idx];
						id[idx] = i;
						map[i] = idx;
						offset[i] = off;
						remove_entry(idx);
						open_off += length;
					}
					else
					{
						// replace
						idx = first[0];
						off = 0;
						int old_id = id[idx];
						while (old_id != -1)
						{
							map[old_id] = -1;
							int next_id = next[old_id];
							next[old_id] = -1;
							old_id = next_id;
						}
						next[i] = -1;
						id[idx] = i;
						map[i] = idx;
						offset[i] = off;
						remove_first(idx);
						open_idx = idx;
						open_off = length;
					}
					push_last(idx);
					cmiss++;
					return false;
				}
				else
				{
					idx = map[i];
					off = offset[i];
					if (priv[idx] == -1)
					{
						priv[idx] = 0;
						remove_entry(idx);
					}
					else
					{
						remove_entry(idx);
						if (priv[idx] == 0)
						{
							priv[idx] = 1;
							if (priv_avail > 0)
							{
								priv_avail--;
							}
							else
							{
								int idx1 = first[1];
								remove_first(idx1);
								priv[idx1] = 0;
								push_last(idx1);
							}
						}
					}
					push_last(idx);
					chit++;
					return true;
				}
			}
		};

		// least frequently used
		class CacheLFU : public lap::CacheLFU
		{
		protected:
			std::vector<int> offset;
			std::vector<int> next;
			int open_idx;
			int open_off;
			int line_length;

		public:
			CacheLFU() : lap::CacheLFU()
			{
				open_idx = -1;
				open_off = -1;
			}
			~CacheLFU() {}

			__forceinline void setSize(int entries, int dim, int length)
			{
				lap::CacheLFU::setSize(entries, dim);
				line_length = length;
				offset.resize(dim);
				next.resize(dim);
				for (int i = 0; i < dim; i++) offset[i] = -1;
				for (int i = 0; i < dim; i++) next[i] = -1;
			}

			__forceinline bool find(int &idx, int &off, int i, int length)
			{
				if (map[i] == -1)
				{
					if ((open_idx >= 0) && (open_off + length <= line_length))
					{
						// append
						idx = open_idx;
						off = open_off;
						next[i] = id[idx];
						id[idx] = i;
						map[i] = idx;
						offset[i] = off;
						open_idx = idx;
						open_off = length;
						count[i]++;
						advance(pos[idx]);
					}
					else
					{
						// replace
						idx = order[0];
						off = 0;
						int old_id = id[idx];
						while (old_id != -1)
						{
							map[old_id] = -1;
							int next_id = next[old_id];
							next[old_id] = -1;
							old_id = next_id;
						}
						next[i] = -1;
						id[idx] = i;
						map[i] = idx;
						offset[i] = off;
						open_idx = idx;
						open_off = length;
						count[i]++;
						advance(0);
					}
					cmiss++;
					return false;
				}
				else
				{
					idx = map[i];
					off = offset[i];
					count[i]++;
					advance(pos[idx]);
					chit++;
					return true;
				}
			}
		};
	}
}
