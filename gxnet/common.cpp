
#include "common.h"

namespace gxnet {

bool gx_is_inner_debug = false;

stdx::element_aligned_tag Aligned = stdx::element_aligned;

DataType gx_inner_product( const DataType * a, const DataType * b, size_t count )
{
	DataType result = 0;

#if 1
	size_t idx = 0;

	for( ; ( idx + 8 * DataSimd::size() - 1 ) < count; idx += 8 * DataSimd::size() ) {
		const DataType * pA = a + idx;
		const DataType * pB = b + idx;

		DataSimd tA0( pA, Aligned );
		tA0 *= DataSimd( pB, Aligned );

		DataSimd tA1( pA + DataSimd::size(), Aligned );
		tA1 *= DataSimd( pB + DataSimd::size(), Aligned );

		DataSimd tA2( pA + 2 * DataSimd::size(), Aligned );
		tA2 *= DataSimd( pB + 2 * DataSimd::size(), Aligned );

		DataSimd tA3( pA + 3 * DataSimd::size(), Aligned );
		tA3 *= DataSimd( pB + 3 * DataSimd::size(), Aligned );

		DataSimd tA4( pA + 4 * DataSimd::size(), Aligned );
		tA4 *= DataSimd( pB + 4 * DataSimd::size(), Aligned );

		DataSimd tA5( pA + 5 * DataSimd::size(), Aligned );
		tA5 *= DataSimd( pB + 5 * DataSimd::size(), Aligned );

		DataSimd tA6( pA + 6 * DataSimd::size(), Aligned );
		tA6 *= DataSimd( pB + 6 * DataSimd::size(), Aligned );

		DataSimd tA7( pA + 7 * DataSimd::size(), Aligned );
		tA7 *= DataSimd( pB + 7 * DataSimd::size(), Aligned );

		tA0 += tA1 + tA2 + tA3 + tA4 + tA5 + tA6 + tA7;

		result += stdx::reduce( tA0, std::plus{} );
	}

	for( ; ( idx + 2 * DataSimd::size() - 1 ) < count; idx += 2 * DataSimd::size() ) {
		const DataType * pA = a + idx;
		const DataType * pB = b + idx;

		DataSimd tA0( pA, Aligned );
		tA0 *= DataSimd( pB, Aligned );

		DataSimd tA1( pA + DataSimd::size(), Aligned );
		tA1 *= DataSimd( pB + DataSimd::size(), Aligned );

		tA0 += tA1;

		result += stdx::reduce( tA0, std::plus{} );
	}

	for( ; ( idx + DataSimd::size() - 1 ) < count; idx += DataSimd::size() ) {
		const DataType * pA = a + idx;
		const DataType * pB = b + idx;

		DataSimd tA0( pA, Aligned );
		tA0 *= DataSimd( pB, Aligned );

		result += stdx::reduce( tA0, std::plus{} );
	}

	for( ; idx < count; idx++ ) result += a[ idx ] * b[ idx ];
#else

	result = std::transform_reduce( a, a + count, b, 0.0 );

	//for( size_t idx = 0; idx < count; idx++ ) result += a[ idx ] * b[ idx ];

#endif

	return result;
}

void gx_vs_product( const DataType * a, const DataType & b, DataType * c, size_t count )
{
	size_t idx = 0;

	DataVector temp( b, DataSimd::size() );
	DataSimd tB( std::begin( temp ), Aligned );

	for( ; ( idx + 8 * DataSimd::size() - 1 ) < count; idx += 8 * DataSimd::size() ) {
		const DataType * pA = a + idx;
		DataType * pC = c + idx;

		DataSimd tA0 = tB * DataSimd( pA, Aligned );
		tA0.copy_to( pC, Aligned );

		DataSimd tA1 = tB * DataSimd( pA + DataSimd::size(), Aligned );
		tA1.copy_to( pC + DataSimd::size(), Aligned );

		DataSimd tA2 = tB * DataSimd( pA + 2 * DataSimd::size(), Aligned );
		tA2.copy_to( pC + 2 * DataSimd::size(), Aligned );

		DataSimd tA3 = tB * DataSimd( pA + 3 * DataSimd::size(), Aligned );
		tA3.copy_to( pC + 3 * DataSimd::size(), Aligned );

		DataSimd tA4 = tB * DataSimd( pA + 4 * DataSimd::size(), Aligned );
		tA4.copy_to( pC + 4 * DataSimd::size(), Aligned );

		DataSimd tA5 = tB * DataSimd( pA + 5 * DataSimd::size(), Aligned );
		tA5.copy_to( pC + 5 * DataSimd::size(), Aligned );

		DataSimd tA6 = tB * DataSimd( pA + 6 * DataSimd::size(), Aligned );
		tA6.copy_to( pC + 6 * DataSimd::size(), Aligned );

		DataSimd tA7 = tB * DataSimd( pA + 7 * DataSimd::size(), Aligned );
		tA7.copy_to( pC + 7 * DataSimd::size(), Aligned );
	}

	for( ; ( idx + 2 * DataSimd::size() - 1 ) < count; idx += 2 * DataSimd::size() ) {
		const DataType * pA = a + idx;
		DataType * pC = c + idx;

		DataSimd tA0 = tB * DataSimd( pA, Aligned );
		tA0.copy_to( pC, Aligned );

		DataSimd tA1 = tB * DataSimd( pA + DataSimd::size(), Aligned );
		tA1.copy_to( pC + DataSimd::size(), Aligned );
	}

	for( ; ( idx + DataSimd::size() - 1 ) < count; idx += DataSimd::size() ) {
		const DataType * pA = a + idx;
		DataType * pC = c + idx;

		DataSimd tA0 = tB * DataSimd( pA, Aligned );

		tA0.copy_to( pC, Aligned );
	}

	for( ; idx < count; idx++ ) c[ idx ] = b * a[ idx ];
}

}; // namespace gxnet;

