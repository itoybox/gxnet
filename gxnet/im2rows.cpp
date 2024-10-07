
#include "im2rows.h"

namespace gxnet {

void Im2Rows :: filters2Rows( const DataVector & src, const Dims & dims, DataMatrix * dest )
{
	Dims rowDims = { dims[ 1 ], dims[ 2 ], dims[ 3 ] };
	size_t rowSize = gx_dims_flatten_size( rowDims );

	if( dest->size() == 0 ) {
		dest->resize( dims[ 0 ] );
		for( auto & vec : *dest )  vec.resize( rowSize );
	}

	MDSpanRO srcMS( src, dims );
	for( size_t f = 0; f < dims[ 0 ]; f++ ) {
		MDSpanRW rowMS( ( *dest )[ f ], rowDims );

		for( size_t c = 0; c < rowDims[ 0 ]; c++ ) {
			for( size_t x = 0; x < rowDims[ 1 ]; x++ ) {
				for( size_t y = 0; y < rowDims[ 2 ]; y++ ) {
					rowMS( c, x, y ) = srcMS( f, c, x, y );
				}
			}
		}
	}
}

void Im2Rows :: deltas2Rows( const DataVector & src, size_t sampleIndex, const Dims & dims,
		DataMatrix * dest )
{
	Dims rowDims = { dims[ 2 ], dims[ 3 ] };
	size_t rowSize = gx_dims_flatten_size( rowDims );

	if( dest->size() == 0 ) {
		dest->resize( dims[ 1 ] );
		for( auto & vec : *dest )  vec.resize( rowSize );
	}

	MDSpanRO srcMS( src, dims );
	for( size_t f = 0; f < dims[ 1 ]; f++ ) {
		MDSpanRW rowMS( ( *dest )[ f ], rowDims );

		for( size_t x = 0; x < rowDims[ 0 ]; x++ ) {
			for( size_t y = 0; y < rowDims[ 1 ]; y++ ) {
				rowMS( x, y ) = srcMS( sampleIndex, f, x, y );
			}
		}
	}
}

void Im2Rows :: rot180Filters2Rows( const DataVector & src, const Dims & dims,
		DataMatrix * rot180 )
{
	Dims rowDims = { dims[ 0 ], dims[ 2 ], dims[ 3 ] };
	size_t rowSize = gx_dims_flatten_size( rowDims );

	if( rot180->size() == 0 ) {
		rot180->resize( dims[ 1 ] );
		for( auto & vec : *rot180 ) vec.resize( rowSize );
	}

	DataVector temp( src.size() );
	rot180Filters( src, dims, &temp );

	MDSpanRO tempMS( temp, dims );

	for( size_t c = 0; c < dims[ 1 ]; c++ ) {
		MDSpanRW rowMS( ( *rot180 )[ c ], rowDims );

		for( size_t f = 0; f < rowDims[ 0 ]; f++ ) {
			for( size_t x = 0; x < rowDims[ 1 ]; x++ ) {
				for( size_t y = 0; y < rowDims[ 2 ]; y++ ) {
					rowMS( f, x, y ) = tempMS( f, c, x, y );
				}
			}
		}
	}

	//Utils::printMatrix( "im2row.rot180", *rot180, false );
}

void Im2Rows :: input2Rows( const MDSpanRO & inMS, size_t sampleIndex, const Dims & filterDims, DataMatrix * dest )
{
	Dims rowDims = { filterDims[ 1 ], filterDims[ 2 ], filterDims[ 3 ] };

	if( dest->size() == 0 ) {
		for( size_t x = 0; x < inMS.dim( 2 ) - filterDims[ 2 ] + 1; x++ ) {
			for( size_t y = 0; y < inMS.dim( 3 ) - filterDims[ 3 ] + 1; y++ ) {
				dest->emplace_back( DataVector( gx_dims_flatten_size( rowDims ) ) );
			}
		}
	}

	DataMatrix::iterator iter = dest->begin();

	for( size_t x = 0; x < inMS.dim( 2 ) - filterDims[ 2 ] + 1; x++ ) {
		for( size_t y = 0; y < inMS.dim( 3 ) - filterDims[ 3 ] + 1; y++ ) {
			MDSpanRW inRowMS( *iter, rowDims );

			for( size_t c = 0; c < inMS.dim( 1 ); c++ ) {
				for( size_t i = 0; i < filterDims[ 2 ]; i++ ) {
					for( size_t j = 0; j < filterDims[ 3 ]; j++ ) {
						inRowMS( c, i, j ) = inMS( sampleIndex, c, x + i, y + j );
					}
				}
			}
			++iter;
		}
	}
}

void Im2Rows :: input2Rows4Gradients( const MDSpanRO & inMS, size_t sampleIndex,
		const Dims & filterDims, DataMatrix * dest )
{
	Dims rowDims = { filterDims[ 2 ], filterDims[ 3 ] };

	if( dest->size() == 0 ) {
		for( size_t c = 0; c < inMS.dim( 1 ); c++ ) {
			for( size_t x = 0; x < inMS.dim( 2 ) - filterDims[ 2 ] + 1; x++ ) {
				for( size_t y = 0; y < inMS.dim( 3 ) - filterDims[ 3 ] + 1; y++ ) {
					dest->emplace_back( DataVector( gx_dims_flatten_size( rowDims ) ) );
				}
			}
		}
	}

	DataMatrix::iterator iter = dest->begin();

	for( size_t c = 0; c < inMS.dim( 1 ); c++ ) {
		for( size_t x = 0; x < inMS.dim( 2 ) - filterDims[ 2 ] + 1; x++ ) {
			for( size_t y = 0; y < inMS.dim( 3 ) - filterDims[ 3 ] + 1; y++ ) {
				MDSpanRW inRowMS( *iter, rowDims );

				for( size_t i = 0; i < filterDims[ 2 ]; i++ ) {
					for( size_t j = 0; j < filterDims[ 3 ]; j++ ) {
						inRowMS( i, j ) = inMS( sampleIndex, c, x + i, y + j );
					}
				}
				++iter;
			}
		}
	}
}

void Im2Rows :: rot180Filters( const DataVector & src, const Dims & dims, DataVector * dest )
{
	MDSpanRO srcMS( src, dims );
	MDSpanRW destMS( *dest, dims );

	dest->resize( gx_dims_flatten_size( dims ) );

	for( size_t f = 0; f < dims[ 0 ]; f++ ) {
		for( size_t c = 0; c < dims[ 1 ]; c++ ) {
			for( size_t i = 0; i < dims[ 2 ]; i++ ) {
				for( size_t j = 0; j < dims[ 3 ]; j++ ) {
					destMS( f, c, dims[ 2 ] - i - 1, dims[ 3 ] - j - 1 ) = srcMS( f, c, i, j );
				}
			}
		}
	}
}


}; // namespace gxnet;

