#include <vector>
#include <iostream>
#include "timer.h"
#include <random>
#include <omp.h>

template <size_t M, size_t N>
struct matrix
{
	std::vector<double> A;

	matrix()
	{
		A.resize(M * N);
	}

	matrix(const matrix& m) :A(m.A)
	{
		std::cout << "copy \n";
	}

	matrix(const std::initializer_list<double>& list)
	{
		A.resize(M * N);
		size_t i = 0;
		for (auto p : list)
			A[i++] = p;
	}

	inline double& operator()(size_t i, size_t j)
	{
		return A[N * i + j];
	}

	inline const double& operator()(size_t i, size_t j) const
	{
		return A[N * i + j];
	}

	matrix<M, N> is_L_U_correct()
	{
		matrix<M, N> res;
		for (size_t i = 0; i < M; ++i)
			for (size_t j = 0; j < N; ++j)
				for (size_t k = 0; k < i + 1 && k < j + 1; ++k)
				{
					if (i == k)
					{
						res(i, j) += (*this)(i, j);
						continue;
					}
					res(i, j) += (*this)(i, k) * (*this)(k, j);
				}
		return res;
	}

	template<size_t b>
	void get_L_2_2(matrix<b, b>& L_2_2, size_t num_of_iteration)
	{
		size_t step = num_of_iteration * b;
		for (size_t k = 0; k < b; k++)
			for (size_t m = 0; m < b; m++)
				L_2_2(k, m) = (*this)(step + k, step + m);
	}

	template<size_t b>
	void get_L_3_2(matrix<M - b, b>& L_3_2, size_t num_of_iteration)
	{
		size_t step = num_of_iteration * b;
		size_t next_step = (1 + num_of_iteration) * b;
		for (size_t i = next_step; i < M; ++i)
			for (size_t j = step; j < next_step; ++j)
				L_3_2(i - next_step, j - step) = (*this)(i, j);
	}

	template<size_t b>
	void get_U_2_3(matrix<b, M - b>& L_2_3, size_t num_of_iteration)
	{
		for (size_t j = b * (num_of_iteration + 1); j < M; ++j)
			for (size_t i = b * num_of_iteration; i < b * (num_of_iteration + 1); ++i)
				L_2_3(i - b * num_of_iteration, j - b * (num_of_iteration + 1)) = (*this)(i, j);
	}

	//вспомоготальный метод, нужный только для расчета U_2_3 (этот метод вызывает объект U_2_3)
	template<size_t b>
	void calculate_U_2_3(const matrix<b, b>& L_2_2, size_t num_of_iteration)
	{
		for (size_t j = 0; j < N - (b * num_of_iteration); ++j)
		{
			for (size_t i = 1; i < b; ++i)
				for (size_t k = 0; k < i; ++k)
					(*this)(i, j) = (*this)(i, j) - (*this)(k, j) * L_2_2(i, k);
		}
	}

	template<size_t b>
	void set_L_2_2(matrix<b, b>& L_2_2, size_t num_of_iteration)
	{
		for (size_t k = 0; k < b; k++)
			for (size_t m = 0; m < b; m++)
				(*this)(num_of_iteration * b + k, num_of_iteration * b + m) = L_2_2(k, m);
	}

	template<size_t b>
	void set_L_3_2(matrix<M - b, b>& L_3_2, size_t num_of_iteration)
	{
		for (size_t i = b * (num_of_iteration + 1); i < M; ++i)
			for (size_t j = b * num_of_iteration; j < b * (num_of_iteration + 1); ++j)
				(*this)(i, j) = L_3_2(i - b * (num_of_iteration + 1), j - b * num_of_iteration);
	}

	template<size_t b>
	void set_U_2_3(matrix<b, M - b>& L_2_3, size_t num_of_iteration)
	{
		for (size_t j = b * (num_of_iteration + 1); j < M; ++j)
			for (size_t i = b * num_of_iteration; i < b * (num_of_iteration + 1); ++i)
				(*this)(i, j) = L_2_3(i - b * num_of_iteration, j - b * (num_of_iteration + 1));
	}

	template<size_t b>
	void set_A_3_3(const matrix<M - b, b>& L_3_2, const matrix<b, M - b>& U_2_3, size_t num_of_iteration)
	{
		for (size_t i = b * (num_of_iteration + 1); i < M; ++i)
			for (size_t k = 0; k < b; ++k)
				for (size_t j = b * (num_of_iteration + 1); j < M; ++j)
					(*this)(i, j) -= L_3_2(i - b * (num_of_iteration + 1), k) * U_2_3(k, j - b * (num_of_iteration + 1));
	}


	template<size_t K, size_t J>
	void get_submatrix(matrix<K, J>& res, size_t i, size_t j) const
	{
		for (size_t k = 0; k < K; k++)
			for (size_t m = 0; m < J; m++)
				res(k, m) = (*this)(i * K + k, j * J + m);
	}

	template<size_t K, size_t J>
	void set_submatrix(matrix<K, J>& res, size_t i, size_t j)
	{
		for (size_t k = 0; k < K; k++)
			for (size_t m = 0; m < J; m++)
				(*this)(i * K + k, j * J + m) = res(k, m);
	}



	template<size_t b, size_t SZ_OF_BLOCK>
	void set_A_3_3_bl(const matrix<M - b, b>& L_3_2, const matrix<b, M - b>& U_2_3, size_t num_of_iteration)
	{
		matrix<SZ_OF_BLOCK, SZ_OF_BLOCK> A;
		matrix<SZ_OF_BLOCK, SZ_OF_BLOCK> B;
		auto count_of_iteration_for_block_dot = (M - b * (num_of_iteration + 1)) / SZ_OF_BLOCK;
		for (size_t i = 0; i < count_of_iteration_for_block_dot; ++i)
			for (size_t k = 0; k < b / SZ_OF_BLOCK; ++k)
				for (size_t j = 0; j < count_of_iteration_for_block_dot; ++j)
				{
					L_3_2.get_submatrix(A, i, k); U_2_3.get_submatrix(B, k, j);
					auto step_i = i * SZ_OF_BLOCK + b * (num_of_iteration + 1);
					auto step_j = j * SZ_OF_BLOCK + b * (num_of_iteration + 1);
					for (size_t m = step_i; m < step_i + SZ_OF_BLOCK; ++m)
						for (size_t s = 0; s < SZ_OF_BLOCK; ++s)
							for (size_t q = step_j; q < step_j + SZ_OF_BLOCK; ++q)
								(*this)(m, q) -= A(m - step_i, s) * B(s, q - step_j);
				}
	}

};


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
void standart_L_U(matrix<M, M>& A)
{
	/*const size_t N = 6;
	matrix<N, N> A = { 2,3,4,6,7,8,
					  7,2,3,5,4,6,
					  8,9,3,5,8,1,
					  9,9,9,3,2,1,
					  7,3,5,8,3,1,
					  9,8,7,6,5,4 };
	std::cout << A << "\n \n";*/
	for (size_t j = 0; j < M; ++j)
	{
		for (size_t i = j + 1; i < M; ++i)     //заполнение столбцa j матрицы L
			A(i, j) = A(i, j) / A(j, j);

		for (size_t i = j + 1; i < M; ++i)
			for (size_t k = j + 1; k < M; ++k)  //заполнение строки i матрицы U
				A(i, k) = A(i, k) - A(j, k) * A(i, j);
	}
	//	std::cout << A << "\n\n";
		//std::cout << A.is_L_U_correct() << "\n \n";
	std::cout << A(456, 345) << "\n";
}

template<size_t M, size_t b>
void block_L_U(matrix<M, M>& A)
{
	//const size_t M = 6;
	//const size_t b = 2;
	//matrix<M, M> A = { 2,3,4,6,7,8,
	//				  7,2,3,5,4,6,
	//				  8,9,3,5,8,1,
	//				  9,9,9,3,2,1,
	//				  7,3,5,8,3,1,
	//				  9,8,7,6,5,4 };
	//std::cout << A << "\n\n";
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
	std::cout << A(456, 345) << "\n";
	//std::cout << A << "\n\n";
	//std::cout << A.is_L_U_correct() << "\n\n";
}

template<size_t M, size_t b>
void block_L_U_block_dot(matrix<M, M>& A)
{
	//const size_t M = 6;
	//const size_t b = 2;
	//matrix<M, M> A = { 2,3,4,6,7,8,
	//				  7,2,3,5,4,6,
	//				  8,9,3,5,8,1,
	//				  9,9,9,3,2,1,
	//				  7,3,5,8,3,1,
	//				  9,8,7,6,5,4 };
	//std::cout << A << "\n\n";
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
		A.set_A_3_3_bl<b, 128>(L_3_2, U_2_3, j);
	}
	std::cout << A(456, 345) << "\n";
	//std::cout << A << "\n\n";
	//std::cout << A.is_L_U_correct() << "\n\n";
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
	const int M = 4096;
	const int b = 256;
	matrix<M, M> A;
	create_matrix(A);
	//matrix<M, M> A = {      2,3,4,6,7,8,
	//						7,2,3,5,4,6,
	//						8,9,3,5,8,1,
	//						9,9,9,3,2,1,
	//						7,3,5,8,3,1,
	//						9,8,7,6,5,4 };
	auto B = A;
	{
		timer t("Block LU");
		block_L_U<M, b>(A);
	}
	A = B;
	{
		timer t("Block LU block dot");
		block_L_U_block_dot<M, b>(A);
	}
	A = B;
	{	timer t("Standart LU");
	standart_L_U(A);
	}
}