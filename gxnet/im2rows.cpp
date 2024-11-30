
#include "im2rows.h"

namespace gxnet {

void Im2Rows :: rot180Filters2Rows( const MDVector & src, MDVector * rot180 )
{
	MDVector temp;
	rot180Filters( src, &temp );

	MDSpanRO tempRO( temp );

	Dims dims = { temp.second[ 1 ], temp.second[ 0 ], temp.second[ 2 ], temp.second[ 3 ] };
	rot180->first.resize( gx_dims_flatten_size( dims ) );
	rot180->second = { dims[ 0 ], gx_dims_flatten_size( dims ) / dims[ 0 ] };

	MDSpanRW rw( std::begin( rot180->first ), dims );

	for( size_t c = 0; c < dims[ 0 ]; c++ ) {
		for( size_t f = 0; f < dims[ 1 ]; f++ ) {
			for( size_t x = 0; x < dims[ 2 ]; x++ ) {
				for( size_t y = 0; y < dims[ 3 ]; y++ ) {
					rw( c, f, x, y ) = tempRO( f, c, x, y );
				}
			}
		}
	}

	//Utils::printMatrix( "im2row.rot180", *rot180, false );
}

void Im2Rows :: input2Rows( const MDSpanRO & inRO, size_t sampleIndex,
		const Dims & filterDims, MDVector * dest )
{
	size_t xMax = inRO.dim( 2 ) - filterDims[ 2 ] + 1;
	size_t yMax = inRO.dim( 3 ) - filterDims[ 3 ] + 1;

	Dims dims = { xMax * yMax, filterDims[ 1 ], filterDims[ 2 ], filterDims[ 3 ] };
	dest->first.resize( gx_dims_flatten_size( dims ) );

	dest->second = { xMax * yMax, gx_dims_flatten_size( filterDims ) / filterDims[ 0 ] };

	MDSpanRW rw( std::begin( dest->first ), dims );

	for( size_t x = 0; x < xMax; x++ ) {
		for( size_t y = 0; y < yMax; y++ ) {
			for( size_t c = 0; c < inRO.dim( 1 ); c++ ) {
				for( size_t i = 0; i < filterDims[ 2 ]; i++ ) {
					for( size_t j = 0; j < filterDims[ 3 ]; j++ ) {
						rw( ( x * yMax + y ), c, i, j ) = inRO( sampleIndex, c, x + i, y + j );
					}
				}
			}
		}
	}
}

void Im2Rows :: input2Rows4Gradients( const MDSpanRO & inRO, size_t sampleIndex,
		const Dims & filterDims, MDVector * dest )
{
	size_t xMax = inRO.dim( 2 ) - filterDims[ 2 ] + 1;
	size_t yMax = inRO.dim( 3 ) - filterDims[ 3 ] + 1;

	Dims dims = { inRO.dim( 1 ), xMax * yMax, filterDims[ 2 ], filterDims[ 3 ] };
	dest->first.resize( gx_dims_flatten_size( dims ) );

	dest->second = { inRO.dim( 1 ) * xMax * yMax, filterDims[ 2 ] * filterDims[ 3 ] };

	MDSpanRW rw( std::begin( dest->first ), dims );

	for( size_t c = 0; c < inRO.dim( 1 ); c++ ) {
		for( size_t x = 0; x < inRO.dim( 2 ) - filterDims[ 2 ] + 1; x++ ) {
			for( size_t y = 0; y < inRO.dim( 3 ) - filterDims[ 3 ] + 1; y++ ) {

				for( size_t i = 0; i < filterDims[ 2 ]; i++ ) {
					for( size_t j = 0; j < filterDims[ 3 ]; j++ ) {
						rw( c, x * yMax + y, i, j ) = inRO( sampleIndex, c, x + i, y + j );
					}
				}
			}
		}
	}
}

void Im2Rows :: rot180Filters( const MDVector & src, MDVector * dest )
{
	dest->second = src.second;
	dest->first.resize( gx_dims_flatten_size( dest->second ) );

	MDSpanRO srcRO( src );
	MDSpanRW destRW( *dest );

	for( size_t f = 0; f < srcRO.dim( 0 ); f++ ) {
		for( size_t c = 0; c < srcRO.dim( 1 ); c++ ) {
			for( size_t i = 0; i < srcRO.dim( 2 ); i++ ) {
				for( size_t j = 0; j < srcRO.dim( 3 ); j++ ) {
					destRW( f, c, srcRO.dim( 2 ) - i - 1, srcRO.dim( 3 ) - j - 1 ) = srcRO( f, c, i, j );
				}
			}
		}
	}
}

}; // namespace gxnet;

