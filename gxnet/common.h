#pragma once

#include <vector>
#include <valarray>
#include <string>
#include <sstream>
#include <iostream>
#include <unordered_map>

#include <assert.h>
#include <string.h>

#include <experimental/simd>
#include <numeric>

namespace gxnet {

extern bool gx_is_inner_debug;

namespace stdx = std::experimental::parallelism_v2;

typedef double DataType;

typedef stdx::native_simd< DataType > DataSimd;

typedef std::valarray< DataType > DataVector;
typedef std::vector< DataVector > DataMatrix;

typedef std::vector< bool > BoolVector;
typedef std::vector< int > IntVector;

typedef std::vector< size_t > Dims;
typedef std::vector< Dims > DimsList;

class SpanRO;

DataType gx_inner_product( const DataType * a, const DataType * b, size_t count );

void gx_vs_product( const DataType * a, const DataType & b, DataType * c, size_t count );

inline void gx_matrix_add( DataMatrix * dest, const DataMatrix & src )
{
	assert( dest->size() == src.size() );
	for( size_t i = 0; i < src.size(); i++ ) ( *dest )[ i ] += src[ i ];
}

inline size_t gx_dims_flatten_size( const Dims & dims )
{
	size_t ret = dims.size() > 0 ? 1 : 0;
	for( auto & dim : dims ) ret = ret * dim;

	return ret;
}

template< typename NumberVector >
std::string gx_vector2string( const NumberVector & vec, const char delim = ',' )
{
	std::ostringstream ret;
	ret.setf( std::ios::scientific, std::ios::floatfield );

	for( size_t i = 0; i < vec.size(); i++ ) {
		if( i > 0 ) ret << delim;
		ret << vec[ i ];
	}

	return ret.str();
}

template< typename NumberVector >
void gx_string2vector( const std::string & buff, NumberVector * vec, const char delim = ',' )
{
	std::stringstream ss( buff );
	std::string token;
	while( std::getline( ss, token, delim ) ) {
		vec->emplace_back( std::stod( token ) );
	}
}

inline void gx_string2valarray( const std::string & buff, DataVector * vec, const char delim = ',' )
{
	std::stringstream ss( buff );
	std::string token;
	for( size_t i = 0; i < vec->size(); i++ ) {
		if( !std::getline( ss, token, delim ) ) break;
		( *vec )[ i ] = std::stod( token );
	}
}

class MDSpanRW {
public:
	MDSpanRW( DataVector & data, const Dims & dims )
			: mData( data ), mDims( dims ) {
	}

	MDSpanRW( DataVector & data )
			: mData( data ) {
	}

	~MDSpanRW() {}

	DataVector & data() { return mData; }

	Dims & dims() { return mDims; }

	size_t dim( size_t index ) const {
		assert( index < mDims.size() );
		return mDims[ index ];
	}

	DataType & operator()( size_t i ) {
		return mData[ i ];
	}

	DataType & operator()( size_t i, size_t j ) {
		assert( mDims.size() == 2 );
		return mData[ i * mDims[ 1 ] + j ];
	}

	DataType & operator()( size_t i, size_t j, size_t k ) {
		assert( mDims.size() == 3 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * i ) + ( mDims[ 2 ] * j ) + k ];
	}

	DataType & operator()( size_t f, size_t c, size_t i, size_t j ) const {
		assert( mDims.size() == 4 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * mDims[ 3 ] * f )
		       + ( mDims[ 2 ] * mDims[ 3 ] * c ) + ( mDims[ 3 ] * i ) + j ];
	}

private:
	DataVector & mData;
	Dims mDims;
};

class MDSpanRO {
public:
	MDSpanRO( const DataVector & data, const Dims & dims )
			: mData( data ), mDims( dims ) {
		//assert( gx_dims_flatten_size( dims ) == data.size() );
	}

	MDSpanRO( MDSpanRW & other )
		: mData( other.data() ), mDims( other.dims() ) {
	}

	~MDSpanRO() {}

	const DataVector & data() const { return mData; };

	const Dims & dims() const { return mDims; }

	size_t dim( size_t index ) const {
		assert( index < mDims.size() );
		return mDims[ index ];
	}

	const DataType & operator()( size_t i ) const {
		return mData[ i ];
	}

	const DataType & operator()( size_t i, size_t j ) const {
		assert( mDims.size() == 2 );
		return mData[ i * mDims[ 1 ] + j ];
	}

	const DataType & operator()( size_t i, size_t j, size_t k ) const {
		assert( mDims.size() == 3 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * i ) + ( mDims[ 2 ] * j ) + k ];
	}

	const DataType & operator()( size_t f, size_t c, size_t i, size_t j ) const {
		assert( mDims.size() == 4 );
		return mData[ ( mDims[ 1 ] * mDims[ 2 ] * mDims[ 3 ] * f )
		       + ( mDims[ 2 ] * mDims[ 3 ] * c ) + ( mDims[ 3 ] * i ) + j ];
	}

private:
	const DataVector & mData;
	const Dims & mDims;
};

}; // namespace gxnet;

