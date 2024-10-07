#pragma once

#include "common.h"

namespace gxnet {

class Im2Rows {
public:
	static void filters2Rows( const DataVector & src, const Dims & dims,
			DataMatrix * dest );

	static void deltas2Rows( const DataVector & src, size_t sampleIndex, const Dims & dims,
			DataMatrix * dest );

	static void rot180Filters2Rows( const DataVector & src, const Dims & dims,
			DataMatrix * rot180 );

	static void input2Rows( const MDSpanRO & inMS, size_t sampleIndex, const Dims & filterDims,
			DataMatrix * dest );

	static void input2Rows4Gradients( const MDSpanRO & inMS, size_t sampleIndex,
			const Dims & filterDims, DataMatrix * dest );

	static void rot180Filters( const DataVector & src,
			const Dims & dims, DataVector * dest );
};


}; // namespace gxnet;

