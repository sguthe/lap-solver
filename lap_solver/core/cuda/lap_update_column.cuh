#pragma once

namespace lap
{
	namespace cuda
	{
		__global__ void setColInactive_kernel(char *colactive, int jmin)
		{
			colactive[jmin] = 0;
		}

		template <class SC>
		__global__ void setColInactive_kernel(char *colactive, int jmin, SC *v_jmin, SC *v_in)
		{
			*v_jmin = *v_in;
			colactive[jmin] = 0;
		}

		template <class TC, class TC2, class SC>
		__global__ void setColInactive_kernel(char *colactive, int jmin, TC *tt_jmin, TC2 *tt_in, SC *v_jmin, SC *v_in)
		{
			*tt_jmin = *tt_in;
			*v_jmin = *v_in;
			colactive[jmin] = 0;
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] += dlt;
				v[j] -= dlt + eps;
				total_eps[j] += eps;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int *dst, int *src, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			dst[j] = src[j];
			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int *dst, int *src, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			dst[j] = src[j];
			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] += dlt;
				v[j] -= dlt + eps;
				total_eps[j] += eps;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, int size, int *colsol, int csol)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;
			if (j == 0) *colsol = csol;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				v[j] -= dlt;
			}
		}

		template <class SC>
		__global__ void updateColumnPrices_kernel(char *colactive, SC min, SC *v, SC *d, SC *total_d, SC *total_eps, SC eps, int size, int *colsol, int csol)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;
			if (j == 0) *colsol = csol;

			if (colactive[j] == 0)
			{
				SC dlt = min - d[j];
				total_d[j] += dlt;
				v[j] -= dlt + eps;
				total_eps[j] += eps;
			}
		}

		template <class SC>
		__global__ void markedSkippedColumns_kernel(char *colactive, SC min, int jmin, int *colsol, SC *d, int f, int dim, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			// ignore any columns assigned to virtual rows
			if ((j == jmin) || ((f >= dim) && (colsol[j] >= dim) && (d[j] <= min)))
			{
				colactive[j] = 0;
			}
		}

		template <class SC>
		__global__ void markedSkippedColumnsUpdate_kernel(char *colactive, SC min, int jmin, int *colsol, SC *d, int i, int dim, int size)
		{
			int j = threadIdx.x + blockIdx.x * blockDim.x;
			if (j >= size) return;

			// ignore any columns assigned to virtual rows
			if ((j == jmin) || ((i >= dim) && (colactive[j] == 1) && (colsol[j] >= dim) && (d[j] <= min)))
			{
				colactive[j] = 0;
			}
		}
	}
}
