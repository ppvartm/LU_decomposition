#include <vector>
#include <iostream>
#include "timer.h"
#include "matrix.h"
#include <random>
#include <omp.h>




template<size_t M, size_t N, size_t K>
matrix<M, K>& dot(const matrix<M, N>& A, const matrix<N, K>& B, matrix<M, K>& res)
{
	for (size_t i = 0; i < M; ++i)
		for (size_t k = 0; k < N; ++k)
			for (size_t j = 0; j < K; ++j)
				res(i, j) += A(i, k) * B(k, j);
	return res;
}


template<size_t M, size_t N>
std::ostream& operator<<(std::ostream& os, const matrix<M, N>& m)
{
	for (size_t i = 0; i < M; ++i)
	{
		os << "\n";
		for (size_t j = 0; j < N; ++j)
			os << m(i, j) << "  ";
	}
	return os;
}


template<size_t b, size_t K>
void L_U_for_rect(matrix<b, b>& L_2_2, matrix<K, b>& L_3_2)
{
	size_t min_M_N = std::min(K + b - 1, b);
	for (size_t j = 0; j < min_M_N; ++j)
	{
		for (size_t i = j + 1; i < b; ++i)
			L_2_2(i, j) = L_2_2(i, j) / L_2_2(j, j);
		for (size_t i = b; i < K + b; ++i)
			L_3_2(i - b, j) = L_3_2(i - b, j) / L_2_2(j, j);


		if (j < b)
		{
			for (size_t i = j + 1; i < b; ++i)
				for (size_t k = j + 1; k < b; ++k)
					L_2_2(i, k) = L_2_2(i, k) - L_2_2(j, k) * L_2_2(i, j);

			for (size_t i = b; i < K + b; ++i)
				for (size_t k = j + 1; k < b; ++k)
					L_3_2(i - b, k) = L_3_2(i - b, k) - L_2_2(j, k) * L_3_2(i - b, j);
		}
	}
}





template<size_t M>
void parallel_standart_L_U(matrix<M, M>& A)
{
		for (size_t j = 0; j < M; ++j)
		{
           #pragma omp parallel
			{
              #pragma omp for
				for (int i = j + 1; i < M; ++i)     //заполнение столбцa j матрицы L
					A(i, j) /= A(j, j);

               #pragma omp for
				for (int i = j + 1; i < M; ++i)
					for (int k = j + 1; k < M; ++k)  //заполнение строки i матрицы U
						A.A[M * i + k] = A.A[M * i + k] - A.A[M * j + k] * A.A[M * i + j];
			}
		}
//		std::cout << A(456, 345) << "\n";
}

template<size_t M>
void standart_L_U(matrix<M, M>& A)
{
	for (size_t j = 0; j < M; ++j)
	{
			for (int i = j + 1; i < M; ++i)     //заполнение столбцa j матрицы L
				A(i, j) /= A(j, j);

			for (int i = j + 1; i < M; ++i)
				for (int k = j + 1; k < M; ++k)  //заполнение строки i матрицы U
					A.A[M * i + k] = A.A[M * i + k] - A.A[M * j + k] * A.A[M * i + j];
	}
	//std::cout << A(456, 345) << "\n";
}

template<size_t M, size_t b>
void block_L_U(matrix<M, M>& A)
{
	matrix<b, b> L_2_2;
	matrix<M - b, b> L_3_2;
	matrix<b, M - b> U_2_3;

	for (size_t j = 0; j < M / b; ++j)
	{
		A.get_L_2_2(L_2_2, j);
		A.get_L_3_2<b>(L_3_2, j);
		A.get_U_2_3(U_2_3, j);
		L_U_for_rect(L_2_2, L_3_2);
		U_2_3.calculate_U_2_3(L_2_2, j);
		A.set_L_2_2(L_2_2, j);
		A.set_L_3_2<b>(L_3_2, j);
		A.set_U_2_3(U_2_3, j);
		A.set_A_3_3(L_3_2, U_2_3, j);
	}
	//std::cout << A(456, 345) << "\n";
}

template<size_t M, size_t b, size_t sz_dot>
void block_L_U_block_dot(matrix<M, M>& A)
{
	
	matrix<b, b> L_2_2;
	matrix<M - b, b> L_3_2;
	matrix<b, M - b> U_2_3;

	for (size_t j = 0; j < M / b; ++j)
	{
		A.get_L_2_2(L_2_2, j);
		A.get_L_3_2<b>(L_3_2, j);
		A.get_U_2_3(U_2_3, j);
		L_U_for_rect(L_2_2, L_3_2);
		U_2_3.calculate_U_2_3(L_2_2, j);
		A.set_L_2_2(L_2_2, j);
		A.set_L_3_2<b>(L_3_2, j);
		A.set_U_2_3(U_2_3, j);
		A.set_A_3_3_bl<b, sz_dot>(L_3_2, U_2_3, j);
	}
	//std::cout << A(456, 345) << "\n";
	
}

template<size_t M, size_t b>
void parallel_block_L_U(matrix<M, M>& A)
{
	matrix<b, b> L_2_2;
	matrix<M - b, b> L_3_2;
	matrix<b, M - b> U_2_3;

	size_t min_M_N = std::min(M - 1, b);

	for (size_t j = 0; j < M / b; ++j)
	{
	#pragma omp parallel shared(L_2_2, L_3_2, U_2_3)
	  {
		//GET L22
        #pragma omp for
		for (int k = 0; k < b; k++)
			for (int m = 0; m < b; m++)
				L_2_2(k, m) = A(j*b + k, j * b + m);
		
		// GET L32
        #pragma omp for
		for (int k = (1 + j) * b; k < M; ++k)
			for (int m = j * b; m < (1 + j) * b; ++m)
				L_3_2(k - (1 + j) * b, m - j * b) = A(k, m);
		
		// GET U23
        #pragma omp for
		for (int k = b * (j + 1); k < M; ++k)
			for (int m = b * j; m < b * (j + 1); ++m)
				U_2_3(m - b * j, k - b * (j + 1)) = A(m, k);
		


		//LU L22 L32

		for (int p = 0; p < min_M_N; ++p)
		{
            #pragma omp for
			for (int m = p + 1; m < b; ++m)
				L_2_2(m, p) = L_2_2(m, p) / L_2_2(p, p);
            #pragma omp for
			for (int m = b; m < M; ++m)
				L_3_2(m - b, p) = L_3_2(m - b, p) / L_2_2(p, p);


			if (p < b)
			{
                #pragma omp for
				for (int m = p + 1; m < b; ++m)
					for (int k = p + 1; k < b; ++k)
						L_2_2(m, k) = L_2_2(m, k) - L_2_2(p, k) * L_2_2(m, p);
                #pragma omp for
				for (int m = b; m < M; ++m)
					for (int k = p + 1; k < b; ++k)
						L_3_2(m - b, k) = L_3_2(m - b, k) - L_2_2(p, k) * L_3_2(m - b, p);
			}
		}

		// CALCULATE U23
		{
			for (int p = 0; p < M - b -(b * j); ++p)
			{
				 #pragma omp for
				for (int m = 1; m < b; ++m)
					for (int k = 0; k < m; ++k)
						U_2_3(m, p) = U_2_3(m, p) - U_2_3(k, p) * L_2_2(m, k);
			}
		}

		
		//SET L22
        #pragma omp for
		for (int k = 0; k < b; k++)
			for (int m = 0; m < b; m++)
				A(j * b + k, j * b + m) = L_2_2(k, m);

		// SET L32
        #pragma omp for
		for (int k = (1 + j) * b; k < M; ++k)
			for (int m = j * b; m < (1 + j) * b; ++m)
				 A(k, m) = L_3_2(k - (1 + j) * b, m - j * b);

		// SET U23
       #pragma omp for
		for (int k = b * (j + 1); k < M; ++k)
			for (int m = b * j; m < b * (j + 1); ++m)
				  A(m, k) = U_2_3(m - b * j, k - b * (j + 1));

		// SET A33
        #pragma omp for
		for (int m = b * (j + 1); m < M; ++m)
			for (int k = 0; k < b; ++k)
				for (int p = b * (j + 1); p < M; ++p)
					A(m, p) -= L_3_2(m - b * (j + 1), k) * U_2_3(k, p - b * (j + 1));
	  }
	

    }

	//std::cout << A(456, 345) << "\n";
//	std::cout << A.is_L_U_correct() << "\n";
}


template<size_t M>
void create_matrix(matrix<M, M>& matrix)
{
	for (size_t i = 0; i < M * M; ++i)
	{
		matrix.A[i] = rand() % 100;
	}
}




int main()
{
	const int M = 2064;
	const int b = 256;
	const int sz_dot = 32;
	matrix<M, M> A;
	create_matrix(A);
	auto B = A;
	/*matrix<M, M> A = {      2,3,4,6,7,8,
							7,2,3,5,4,6,
							8,9,3,5,8,1,
							9,9,9,3,2,1,
							7,3,5,8,3,1,
							9,8,7,6,5,4 };*/
	/*
	{
		timer t("Block LU");
		block_L_U<M, b>(A);
	}
	A = B;
	{
		timer t("Block LU block dot");
		block_L_U_block_dot<M, b>(A);
	}
	A = B;*/
	

	//parallel_block_L_U<M,2>(A);
 {
	   timer t("Standart LU");
	
	   standart_L_U(A);
	}
	A = B;
	{
		timer t("Parallel standart LU");

		parallel_standart_L_U(A);
	}
	A = B;
	{
		timer t("Blok LU/ block size = 256");

		block_L_U<M,b>(A);	
	}
	A = B;
	{
		timer t("Blok LU with block dot/ block size = 256");

		block_L_U_block_dot<M, b, sz_dot>(A);
	}
	A = B;
	{
		timer t("Parallel blok LU/ block size = 256");

		parallel_block_L_U<M, b>(A);
	}
	const int b2 = 128;
	A = B;
	{
		timer t("Blok LU/ block size = 128");

		block_L_U<M, b2>(A);
	}
	A = B;
	{
		timer t("Blok LU with block dot/ block size = 128");

		block_L_U_block_dot<M, b2, sz_dot>(A);
	}
	A = B;
	{
		timer t("Parallel blok LU/ block size = 128");

		parallel_block_L_U<M, b2>(A);
	}
	const int b3 = 64;
	A = B;
	{
		timer t("Blok LU/ block size = 64");

		block_L_U<M, b3>(A);
	}
	A = B;
	{
		timer t("Blok LU with block dot/ block size = 64");

		block_L_U_block_dot<M, b3, sz_dot>(A);
	}
	A = B;
	{
		timer t("Parallel blok LU/ block size = 64");

		parallel_block_L_U<M, b3>(A);
	}
	const int b4 = 32;
	A = B;
	{
		timer t("Blok LU/ block size = 32");

		block_L_U<M, b4>(A);
	}
	A = B;
	{
		timer t("Blok LU with block dot/ block size = 32");

		block_L_U_block_dot<M, b4, sz_dot>(A);
	}
	A = B;
	{
		timer t("Parallel blok LU/ block size = 32");

		parallel_block_L_U<M, b4>(A);
	}
	
}