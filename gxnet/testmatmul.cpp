
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>

#ifdef ENABLE_EIGEN
#include <Eigen/Eigen>
#endif

#include "common.h"

using namespace gxnet;

DataType myrandom()
{
	static std::random_device rd;
	static std::mt19937 gen( rd() );
 
	static std::normal_distribution< DataType > dist( 0.0, 1.0 );
 
	return dist( gen );
}

// The following source code is copied from:
// https://lemire.me/blog/2024/06/13/rolling-your-own-fast-matrix-multiplication-loop-order-and-vectorization/

template <typename T> 
struct Matrix {
  Matrix(size_t rows, size_t cols) : 
  data(new T[rows * cols]), rows(rows), cols(cols) {} 

  T &operator()(size_t i, size_t j) { 
      return data.get()[i * cols + j]; 
  } 
  
  const T &operator()(size_t i, size_t j) const { 
      return data.get()[i * cols + j]; 
  }

  std::unique_ptr<T[]> data; size_t rows; size_t cols; 
};


template <typename T>
void multiply_ikj(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t i = 0; i < a.rows; i++) {
    for (size_t k = 0; k < a.cols; k++) {
      for (size_t j = 0; j < b.cols; j++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_ijk(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < b.cols; j++) {
      for (size_t k = 0; k < a.cols; k++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_kij(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t k = 0; k < a.cols; k++) {
    for (size_t i = 0; i < a.rows; i++) {
      for (size_t j = 0; j < b.cols; j++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_kji(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t k = 0; k < a.cols; k++) {
    for (size_t j = 0; j < b.cols; j++) {
      for (size_t i = 0; i < a.rows; i++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_jki(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t j = 0; j < b.cols; j++) {
    for (size_t k = 0; k < a.cols; k++) {
      for (size_t i = 0; i < a.rows; i++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_jik(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
  for (size_t j = 0; j < b.cols; j++) {
    for (size_t i = 0; i < a.rows; i++) {
      for (size_t k = 0; k < a.cols; k++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply_eigen_rm(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
#ifdef ENABLE_EIGEN
  Eigen::Map< Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >
      mpA( a.data.get(), a.rows, a.cols );

  Eigen::Map< Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >
      mpB( b.data.get(), b.rows, b.cols ),
      mpC( c.data.get(), c.rows, c.cols );

  mpC = mpA * mpB;
#endif
}

template <typename T>
void multiply_eigen_cm(const Matrix<T> &a, const Matrix<T> &b, Matrix<T> &c) {
#ifdef ENABLE_EIGEN
  Eigen::Map< Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor > >
      mpA( a.data.get(), a.rows, a.cols ), mpB( b.data.get(), b.rows, b.cols ),
      mpC( c.data.get(), c.rows, c.cols );

  mpC = mpA * mpB;
#endif
}

typedef Matrix< DataType > DMatrix;
const int MAX_COUNT = 1000;

void test( int type, const DMatrix & a, const DMatrix & b, DMatrix & c )
{
	switch( type ) {
		case 0:
			multiply_ikj( a, b, c );
			break;
		case 1:
			multiply_ijk( a, b, c );
			break;
		case 2:
			multiply_kij( a, b, c );
			break;
		case 3:
			multiply_kji( a, b, c );
			break;
		case 4:
			multiply_jki( a, b, c );
			break;
		case 5:
			multiply_jik( a, b, c );
			break;
		case 6:
			multiply_eigen_rm( a, b, c );
			break;
		case 7:
			multiply_eigen_cm( a, b, c );
			break;
	}
}

void test0()
{
	DMatrix a( MAX_COUNT, MAX_COUNT ), b( MAX_COUNT, MAX_COUNT ), c( MAX_COUNT, MAX_COUNT );

	for( int i = 0; i < MAX_COUNT; i++ ) {
		for( int j = 0; j < MAX_COUNT; j++ ) {
			a( i, j ) = myrandom();
			b( i, j ) = myrandom();
		}
	}

	for( int i = 0; i < 8; i++ ) {
		std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();	

		for( int j = 0; j < 10; j++ ) test( i, a, b, c );

		std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();	

		auto timeSpan = std::chrono::duration_cast<std::chrono::milliseconds>( endTime - beginTime );

		printf( "#%d Elapsed time: %.3f\n", i, timeSpan.count() / 1000.0 );
	}

}

int main()
{
	test0();

	return 0;
}

