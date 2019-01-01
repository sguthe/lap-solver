#pragma once

#include <chrono>
#include <sstream>
#include <iostream>
#include <cstring>
#ifndef LAP_QUIET
#include <deque>
#include <mutex>
#endif

namespace lap
{
#ifndef LAP_QUIET
	class AllocationLogger
	{
		std::deque<void *> allocated;
		std::deque<unsigned long long> size;
		std::deque<char *> alloc_file;
		std::deque<int> alloc_line;
		unsigned long long peak;
		unsigned long long current;
		std::mutex lock;
	private:
		std::string commify(unsigned long long n)
		{
			std::string s;
			int cnt = 0;
			do
			{
				s.insert(0, 1, char('0' + n % 10));
				n /= 10;
				if (++cnt == 3 && n)
				{
					s.insert(0, 1, ',');
					cnt = 0;
				}
			} while (n);
			return s;
		}

	public:
		AllocationLogger() { peak = current = (unsigned long long)0; }
		~AllocationLogger() {}
		void destroy()
		{
			lapInfo << "Peak memory usage:" << commify(peak) << " bytes" << std::endl;
			if (allocated.empty()) return;
			lapInfo << "Memory leak list:" << std::endl;
			while (!allocated.empty())
			{
				lapInfo << "  leaked " << commify(size.front()) << " bytes at " << std::hex << allocated.front() << std::dec << ": " << alloc_file.front() << ":" << alloc_line.front() << std::endl;
				size.pop_front();
				allocated.pop_front();
				alloc_file.pop_front();
				alloc_line.pop_front();
			}
		}

		template <class T>
		void free(T a)
		{
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_BEBUG
#pragma omp critical
			lapDebug << "Freeing memory at " << std::hex << a << std::dec << std::endl;
#endif
#endif
			std::lock_guard<std::mutex> guard(lock);
			for (unsigned long long i = 0; i < allocated.size(); i++)
			{
				if ((void *)a == allocated[i])
				{
					current -= size[i];
					allocated[i] = allocated.back();
					allocated.pop_back();
					size[i] = size.back();
					size.pop_back();
					alloc_line[i] = alloc_line.back();
					alloc_line.pop_back();
					alloc_file[i] = alloc_file.back();
					alloc_file.pop_back();
					return;
				}
			}
		}

		template <class T>
		void alloc(T a, unsigned long long s, const char *file, const int line)
		{
#ifdef LAP_DEBUG
#ifndef LAP_NO_MEM_BEBUG
#pragma omp critical
			lapDebug << "Allocating " << s * sizeof(T) << " bytes at " << std::hex << a << std::dec << " \"" << file << ":" << line << std::endl;
#endif
#endif
			std::lock_guard<std::mutex> guard(lock);
			current += s;
			peak = std::max(peak, current);
			allocated.push_back((void *)a);
			size.push_back(s);
			alloc_file.push_back((char *)file);
			alloc_line.push_back(line);
		}
	};

	static AllocationLogger allocationLogger;
#endif

	template <typename T>
	void alloc(T * &ptr, unsigned long long width, const char *file, const int line)
	{
		ptr = new T[width]; // this one is allowed
#ifndef LAP_QUIET
		allocationLogger.alloc(ptr, width, file, line);
#endif
	}

	template <typename T>
	void free(T *&ptr)
	{
		if (ptr == (T *)NULL) return;
#ifndef LAP_QUIET
		allocationLogger.free(ptr);
#endif
		delete[] ptr; // this one is allowed
		ptr = (T *)NULL;
	}

	std::string getTimeString(long long ms)
	{
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
		long long min = sec / 60;
		sec -= min * 60;
		long long hrs = min / 60;
		min -= hrs * 60;
#if defined (_MSC_VER)
		sprintf_s(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#else
		sprintf(time, "%3d:%02d:%02d.%03d", (int)hrs, (int)min, (int)sec, (int)ms);
#endif

		return std::string(time);
	}

	std::string getSecondString(long long ms)
	{
		char time[256];
		long long sec = ms / 1000;
		ms -= sec * 1000;
#if defined (_MSC_VER)
		sprintf_s(time, "%d.%03d", (int)sec, (int)ms);
#else
		sprintf(time, "%d.%03d", (int)sec, (int)ms);
#endif

		return std::string(time);
	}

	template <class TP, class OS>
	void displayTime(TP &start_time, const char *msg, OS &lapStream)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		lapStream << getTimeString(ms) << ": " << msg << " (" << getSecondString(ms) << "s)" << std::endl;
	}

	template <class TP>
	int displayProgress(TP &start_time, int &elapsed, int completed, int target_size, const char *msg = 0, int iteration = -1, bool display = false)
	{
		if (completed == target_size) display = true;

#ifndef LAP_DEBUG
		if (!display) return 0;
#endif

		auto end_time = std::chrono::high_resolution_clock::now();
		long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

#ifdef LAP_DEBUG
		if ((!display) && (elapsed * 10000 < ms))
		{
			elapsed = (int)((ms + 10000ll) / 10000ll);
			lapDebug << getTimeString(ms) << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) lapDebug << " iteration = " << iteration;
			if (msg != 0) lapDebug << msg;
			lapDebug << std::endl;
			return 2;
		}

		if (display)
#endif
		{
			elapsed = (int)((ms + 10000ll) / 10000ll);
			lapInfo << getTimeString(ms) << ": solving " << completed << "/" << target_size;
			if (iteration >= 0) lapInfo << " iteration = " << iteration;
			if (msg != 0) lapInfo << msg;
			lapInfo << std::endl;
			return 1;
		}
#ifdef LAP_DEBUG
		return 0;
#endif
	}

	template <class TC, class I>
	TC guessEpsilon(int x_size, int y_size, I& iterator, int step)
	{
		TC epsilon(0);
		for (int x = 0; x < x_size; x += step)
		{
			const TC *tt = iterator.getRow(x);
			TC min_cost, max_cost;
			min_cost = max_cost = tt[0];
			for (int y = 1; y < y_size; y++)
			{
				TC cost_l = tt[y];
				min_cost = std::min(min_cost, cost_l);
				max_cost = std::max(max_cost, cost_l);
			}
			epsilon += max_cost - min_cost;
		}
		return (epsilon / TC(10 * (x_size + step - 1) / step));
	}

#if defined(__GNUC__)
#define __forceinline \
        __inline__ __attribute__((always_inline))
#endif

	template <class SC, class I>
	__forceinline void initDistance(int f, int dim, int dim2, I &iterator, SC &min, int &jmin, char *colactive, int *pred, int *colsol, SC *d, SC *v)
	{
		min = std::numeric_limits<SC>::max();
		jmin = dim2;
		if (f < dim)
		{
			auto tt = iterator.getRow(f);
			for (int j = 0; j < dim2; j++)
			{
				colactive[j] = 1;
				pred[j] = f;
				SC h = d[j] = tt[j] - v[j];
				if (h < min)
				{
					// better
					jmin = j;
					min = h;
				}
				else if (h == min)
				{
					// same, do only update if old was used and new is free
					if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
				}
			}
		}
		else
		{
			for (int j = 0; j < dim2; j++)
			{
				colactive[j] = 1;
				pred[j] = f;
				SC h = d[j] = -v[j];
				if (h < min)
				{
					// better
					jmin = j;
					min = h;
				}
				else if (h == min)
				{
					// same, do only update if old was used and new is free
					if ((colsol[jmin] >= 0) && (colsol[j] < 0)) jmin = j;
				}
			}
		}
	}

	template <class SC, class I>
	__forceinline void updateDistance(int i, int dim, int dim2, I &iterator, SC min, int jmin, SC &min_n, int &jmin_n, char *colactive, int *pred, int *colsol, SC *d, SC *v)
	{
		jmin_n = dim2;
		min_n = std::numeric_limits<SC>::max();
		if (i < dim)
		{
			auto tt = iterator.getRow(i);
			SC h2 = tt[jmin] - v[jmin] - min;
			for (int j = 0; j < dim2; j++)
			{
				if (colactive[j] != 0)
				{
					SC v2 = tt[j] - v[j] - h2;
					SC h = d[j];
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = v2;
						h = v2;
					}
					if (h < min_n)
					{
						// better
						jmin_n = j;
						min_n = h;
					}
					else if (h == min_n)
					{
						// same, do only update if old was used and new is free
						if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
					}
				}
			}
		}
		else
		{
			SC h2 = -v[jmin] - min;
			for (int j = 0; j < dim2; j++)
			{
				if (colactive[j] != 0)
				{
					SC v2 = -v[j] - h2;
					SC h = d[j];
					if (v2 < h)
					{
						pred[j] = i;
						d[j] = v2;
						h = v2;
					}
					if (h < min_n)
					{
						// better
						jmin_n = j;
						min_n = h;
					}
					else if (h == min_n)
					{
						// same, do only update if old was used and new is free
						if ((colsol[jmin_n] >= 0) && (colsol[j] < 0)) jmin_n = j;
					}
				}
			}
		}
	}

	__forceinline void dijkstraCheck(int &endofpath, bool &unassignedfound, int jmin, int *colsol, char *colactive, int *colcomplete, int &completecount)
	{
		colactive[jmin] = 0;
		colcomplete[completecount++] = jmin;
		if (colsol[jmin] < 0)
		{
			endofpath = jmin;
			unassignedfound = true;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(int *colcomplete, int completecount, SC min, SC *v, SC *d)
	{
		for (int i = 0; i < completecount; i++)
		{
			int j1 = colcomplete[i];
			v[j1] = v[j1] + d[j1] - min;
		}
	}

	template <class SC>
	__forceinline void updateColumnPrices(int *colcomplete, int completecount, SC min, SC *v, SC *d, SC eps, SC &total, unsigned long long &count)
	{
		for (int i = 0; i < completecount; i++)
		{
			int j1 = colcomplete[i];
			v[j1] = v[j1] + d[j1] - min - eps;
			total += d[j1] - min;
		}
		count += completecount;
	}

	__forceinline void resetRowColumnAssignment(int &endofpath, int f, int *pred, int *rowsol, int *colsol)
	{
		int i;
		do
		{
			i = pred[endofpath];
			colsol[endofpath] = i;
			int j1 = endofpath;
			endofpath = rowsol[i];
			rowsol[i] = j1;
		} while (i != f);
	}

	template <class SC, class CF, class I>
	void solve(int dim, int dim2, CF &costfunc, I &iterator, int *rowsol)

		// input:
		// dim        - problem size
		// costfunc - cost matrix
		// findcost   - searching cost matrix

		// output:
		// rowsol     - column assigned to row in solution
		// colsol     - row assigned to column in solution
		// u          - dual variables, row reduction numbers
		// v          - dual variables, column reduction numbers

	{
#ifndef LAP_QUIET
		auto start_time = std::chrono::high_resolution_clock::now();

		long long total_hit = 0LL;
		long long total_miss = 0LL;

		long long total_rows = 0LL;
		long long total_virtual = 0LL;

		int elapsed = -1;
#endif

		int  *pred;
		int  endofpath;
		char *colactive;
		int *colcomplete;
		int completecount;
		SC *d;
		int *colsol;
		SC *v;
		SC *v_old;

#ifdef LAP_DEBUG
		std::vector<SC *> v_list;
		std::vector<SC> eps_list;
#endif

		lapAlloc(colactive, dim2, __FILE__, __LINE__);
		lapAlloc(colcomplete, dim2, __FILE__, __LINE__);
		lapAlloc(d, dim2, __FILE__, __LINE__);
		lapAlloc(pred, dim2, __FILE__, __LINE__);
		lapAlloc(v, dim2, __FILE__, __LINE__);
		lapAlloc(v_old, dim2, __FILE__, __LINE__);
		lapAlloc(colsol, dim2, __FILE__, __LINE__);

		// this is the upper bound
		SC epsilon = costfunc.getInitialEpsilon();
		SC epsilon_lower = epsilon / SC(dim2);

		SC last_avg = SC(0);
		SC last_cost = SC(0);
		bool first = true;
		bool allow_reset = true;

		memset(v, 0, dim2 * sizeof(SC));
		memset(v_old, 0, dim2 * sizeof(SC));

		while (epsilon >= SC(0))
		{
			SC total = SC(0);
			unsigned long long count = 0ULL;
			if (epsilon > SC(0))
			{
				if (epsilon < SC(2) * epsilon_lower) epsilon = SC(0);
				else
				{
					if (!first)
					{
						if ((allow_reset) && (-last_avg <= SC(0.1) * epsilon))
						{
#ifdef LAP_DEBUG
							lapDebug << "modification mostly based on epsilon -> reverting v." << std::endl;
#endif
							memset(v, 0, dim2 * sizeof(SC));
							if (last_avg == SC(0.0)) epsilon *= SC(0.1);
							else epsilon = -last_avg;
						}
						else
						{
							SC cur_cost = cost<SC, CF>(dim, dim2, costfunc, rowsol);
							if (!allow_reset)
							{
								// last_cost is already valid so revert if cost increased
#ifdef LAP_DEBUG
								if (cur_cost > last_cost) lapDebug << "cost increased -> reverting v." << std::endl;
#endif
								if (cur_cost > last_cost) memcpy(v, v_old, dim2 * sizeof(SC));
								else memcpy(v_old, v, dim2 * sizeof(SC));
							}
							else memcpy(v_old, v, dim2 * sizeof(SC));
							last_cost = cur_cost;
							epsilon = std::max(SC(0.25) * epsilon, SC(0.5) * (epsilon + last_avg));
							allow_reset = false;
						}
						if (epsilon < epsilon_lower) epsilon = epsilon_lower;
					}
				}
			}
			SC eps = epsilon;
#ifndef LAP_QUIET
			{
				std::stringstream ss;
				ss << "eps = " << eps;
				const std::string tmp = ss.str();
				displayTime(start_time, tmp.c_str(), lapInfo);
			}
#endif
			// this is to ensure termination of the while statement
			if (epsilon == SC(0)) epsilon = SC(-1.0);
			memset(rowsol, -1, dim2 * sizeof(int));
			memset(colsol, -1, dim2 * sizeof(int));
			int jmin, jmin_n;
			SC min, min_n;
			bool unassignedfound;

#ifndef LAP_QUIET
			int old_complete = 0;
#endif

			// AUGMENT SOLUTION for each free row.
#ifndef LAP_QUIET
			displayProgress(start_time, elapsed, 0, dim2, " rows");
#endif
			for (int f = 0; f < dim2; f++)
			{
#ifndef LAP_QUIET
				if (f < dim)
				{
					total_rows++;
				}
				else
				{
					total_virtual++;
				}
#endif

				unassignedfound = false;
				completecount = 0;

				// Dijkstra search
				initDistance(f, dim, dim2, iterator, min, jmin, colactive, pred, colsol, d, v);
				dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);

				while (!unassignedfound)
				{
					// update 'distances' between freerow and all unscanned columns, via next scanned column.
					int i = colsol[jmin];
#ifndef LAP_QUIET
					if (i < dim)
					{
						total_rows++;
					}
					else
					{
						total_virtual++;
					}
#endif
					updateDistance(i, dim, dim2, iterator, min, jmin, min_n, jmin_n, colactive, pred, colsol, d, v);
					min = std::max(min, min_n);
					jmin = jmin_n;
					dijkstraCheck(endofpath, unassignedfound, jmin, colsol, colactive, colcomplete, completecount);
				}

				// update column prices. can increase or decrease
				if (eps > SC(0))
				{
					updateColumnPrices(colcomplete, completecount, min, v, d, eps, total, count);
				}
				else
				{
					updateColumnPrices(colcomplete, completecount, min, v, d);
				}

				// reset row and column assignments along the alternating path.
				resetRowColumnAssignment(endofpath, f, pred, rowsol, colsol);
#ifndef LAP_QUIET
				int level;
				if ((level = displayProgress(start_time, elapsed, f + 1, dim2, " rows")) != 0)
				{
					long long hit, miss;
					iterator.getHitMiss(hit, miss);
					total_hit += hit;
					total_miss += miss;
					if ((hit != 0) || (miss != 0))
					{
						if (level == 1) lapInfo << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
						else lapDebug << "  hit: " << hit << " miss: " << miss << " (" << miss - (f + 1 - old_complete) << " + " << f + 1 - old_complete << ")" << std::endl;
					}
					old_complete = f + 1;
				}
#endif
			}

			if (count > 0) last_avg = total / SC(count);
			else last_avg = SC(0);

#ifdef LAP_DEBUG
			if (eps > SC(0))
			{
				SC *vv;
				lapAlloc(vv, dim2, __FILE__, __LINE__);
				v_list.push_back(vv);
				eps_list.push_back(eps);
				memcpy(v_list.back(), v, sizeof(SC) * dim2);
			}
			else
			{
				int count = (int)v_list.size();
				if (count > 0)
				{
					for (int l = 0; l < count; l++)
					{
						SC dlt(0), dlt2(0);
						for (int i = 0; i < dim2; i++)
						{
							SC diff = v_list[l][i] - v[i];
							dlt += diff;
							dlt2 += diff * diff;
						}
						dlt /= SC(dim2);
						dlt2 /= SC(dim2);
						lapDebug << "iteration = " << l << " eps/mse = " << eps_list[l] << " " << dlt2 - dlt * dlt << " eps/rmse = " << eps_list[l] << " " << sqrt(dlt2 - dlt * dlt) << std::endl;
						delete v_list[l];
					}
				}
			}
#endif
			first = false;

#ifndef LAP_QUIET
			lapInfo << "  rows evaluated: " << total_rows;
			if (total_virtual > 0) lapInfo << " virtual rows evaluated: " << total_virtual;
			lapInfo << std::endl;
			if ((total_hit != 0) || (total_miss != 0)) lapInfo << "  hit: " << total_hit << " miss: " << total_miss << std::endl;
#endif
		}

		// free reserved memory.
		lapFree(pred);
		lapFree(colactive);
		lapFree(colcomplete);
		lapFree(d);
		lapFree(v_old);
		lapFree(v);
		lapFree(colsol);
	}

	// shortcut for square problems
	template <class SC, class CF, class I>
	void solve(int dim, CF &costfunc, I &iterator, int *rowsol)
	{
		solve<SC>(dim, dim, costfunc, iterator, rowsol);
	}

	template <class SC, class CF>
	SC cost(int dim, int dim2, CF &costfunc, int *rowsol)
	{
		SC total = SC(0);
		for (int i = 0; i < dim; i++) total += costfunc.getCost(i, rowsol[i]);
		return total;
	}

	template <class SC, class CF>
	SC cost(int dim, CF &costfunc, int *rowsol)
	{
		return cost<SC, CF>(dim, dim, costfunc, rowsol);
	}
}
