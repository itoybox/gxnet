#pragma once

#include "common.h"

namespace gxnet {

class Im2Rows {
public:
	static void rot180Filters2Rows( const MDVector & src, MDVector * rot180 );

	static void input2Rows( const MDSpanRO & inRO, size_t sampleIndex, const Dims & filterDims,
			MDVector * dest );

	static void input2Rows4Gradients( const MDSpanRO & inRO, size_t sampleIndex,
			const Dims & filterDims, MDVector * dest );

	static void rot180Filters( const MDVector & src, MDVector * dest );
};


}; // namespace gxnet;

