#include "layer.h"
#include "context.h"

#include "utils.h"
#include "activation.h"
#include "optim.h"

#include "im2rows.h"

#include <limits.h>
#include <cstdio>

#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>

namespace gxnet {

BaseLayer :: BaseLayer( int type )
{
	mType = type;
	mActFunc = NULL;

	mIsTraining = false;
}

BaseLayer :: ~BaseLayer()
{
	if( NULL != mActFunc ) delete mActFunc;
}

void BaseLayer :: forward( BaseLayerContext * ctx ) const
{
	calcOutput( ctx );

	if( NULL != mActFunc ) {
		if( gx_is_inner_debug ) Utils::printMDVector( "before.act", ctx->getOutMD() );

		mActFunc->activate( ctx->getOutMD(), &( ctx->getOutMD() ) );

		if( gx_is_inner_debug ) Utils::printMDVector( "after.act", ctx->getOutMD() );
	}

	ctx->getDeltaMD().second = ctx->getOutMD().second;
	ctx->getDeltaMD().first.resize( ctx->getOutMD().first.size() );
}

void BaseLayer :: backward( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	if( NULL != mActFunc ) {
		mActFunc->derivate( ctx->getOutMD(), &( ctx->getDeltaMD() ) );
	}

	if( NULL != inDelta ) backpropagate( ctx, inDelta );
}

BaseLayerContext * BaseLayer :: createCtx() const
{
	return newCtx();
}

int BaseLayer :: getType() const
{
	return mType;
}

void BaseLayer :: setActFunc( ActFunc * actFunc )
{
	if( NULL != mActFunc ) delete mActFunc;
	mActFunc = actFunc;
}

const ActFunc * BaseLayer :: getActFunc() const
{
	return mActFunc;
}

void BaseLayer :: setTraining( bool isTraining )
{
	mIsTraining = isTraining;
}

const Dims & BaseLayer :: getBaseInDims() const
{
	return mBaseInDims;
}

size_t BaseLayer :: getBaseInSize() const
{
	return gx_dims_flatten_size( mBaseInDims );
}

const Dims & BaseLayer :: getBaseOutDims() const
{
	return mBaseOutDims;
}

size_t BaseLayer :: getBaseOutSize() const
{
	return gx_dims_flatten_size( mBaseOutDims );
}

void BaseLayer :: print( bool isDetail ) const
{
	printf( "Type = %d; ActFuncType = %d; BaseInDims = %s; BaseOutDims = %s; \n",
			mType, mActFunc ? mActFunc->getType() : -1,
			gx_vector2string( mBaseInDims ).c_str(),
			gx_vector2string( mBaseOutDims ).c_str() );

	printWeights( isDetail );
}

void BaseLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	/* do nothing */
}

void BaseLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	/* do nothing */
}

////////////////////////////////////////////////////////////

FullConnLayer :: FullConnLayer( const Dims & baseInDims, size_t neuronCount )
	: BaseLayer( eFullConn )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { neuronCount };

	mWeights.resize( neuronCount );
	for( auto & neuron : mWeights ) {
		neuron.resize( gx_dims_flatten_size( mBaseInDims ) );
		for( auto & w : neuron ) w = gx_is_inner_debug ? gx_debug_weight : Utils::random();
	}

	mBiases.resize( neuronCount );
	for( auto & b : mBiases ) b = gx_is_inner_debug ? gx_debug_weight : Utils::random();
}

FullConnLayer :: ~FullConnLayer()
{
}

void FullConnLayer :: printWeights( bool isDetail ) const
{
	if( !isDetail ) return;

	printf( "Weights: Count = %zu; InSize = %zu;\n", mWeights.size(), mWeights[ 0 ].size() );
	for( size_t i = 0; i < mWeights.size() && i < 10; i++ ) {
		printf( "\tNeuron#%zu: WeightCount = %zu, Bias = %.8f\n", i, mWeights[ i ].size(), mBiases[ i ] );
		for( size_t j = 0; j < mWeights[ i ].size() && j < 10; j++ ) {
			printf( "\t\tWeight#%zu: %.8f\n", j, mWeights[ i ][ j ] );
		}

		if( mWeights[ i ].size() > 10 ) printf( "\t\t......\n" );
	}

	if( mWeights.size() > 10 ) printf( "\t......\n" );
}

const DataMatrix & FullConnLayer :: getWeights() const
{
	return mWeights;
}

const DataVector & FullConnLayer :: getBiases() const
{
	return mBiases;
}

void FullConnLayer :: setWeights( const DataMatrix & weights, const DataVector & biases )
{
	mWeights = weights;
	mBiases = biases;
}

void FullConnLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	size_t total = gx_dims_flatten_size( ctx->getInMD().second );

	size_t inSize = mWeights[ 0 ].size();
	size_t sampleCount = total / inSize;

	ctx->getOutMD().second = { sampleCount, mWeights.size() };
	ctx->getOutMD().first.resize( gx_dims_flatten_size( ctx->getOutMD().second ) );

	MDSpanRW outMS( ctx->getOutMD() );

	const DataType * input = std::begin( ctx->getInMD().first );

	for( size_t n = 0; n < sampleCount; n++, input += inSize ) {
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			outMS( n, i ) = gx_inner_product( std::begin( mWeights[ i ] ), input, inSize );
			if( !gx_is_inner_debug )  outMS( n, i ) += mBiases[ i ];
		}
	}

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "input", ctx->getInMD() );
		Utils::printMDVector( "output", ctx->getOutMD() );
	}
}

void FullConnLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	if( NULL == inDelta ) return;

	FullConnLayerContext * ctxImpl = dynamic_cast< FullConnLayerContext * >( ctx );

	size_t total = gx_dims_flatten_size( ctx->getOutMD().second );
	size_t sampleCount = total / mWeights.size();

	Dims inDeltaDims = { sampleCount, mWeights[ 0 ].size() };
	inDelta->resize( gx_dims_flatten_size( inDeltaDims ) );

	MDSpanRW inDeltaMS( *inDelta, inDeltaDims );

	DataVector & temp = ctxImpl->getTempWeights();
	temp.resize( mWeights.size() );

	for( size_t i = 0; i < mWeights[ 0 ].size(); i++ ) {
		for( size_t j = 0; j < mWeights.size(); j++ ) temp[ j ] = mWeights[ j ][ i ];

		const DataType * delta = std::begin( ctx->getDeltaMD().first );
		for( size_t n = 0; n < sampleCount; n++, delta += temp.size() ) {
			inDeltaMS( n, i ) = gx_inner_product( delta, std::begin( temp ), temp.size() );
		}
	}
}

BaseLayerContext * FullConnLayer :: newCtx() const
{
	return new FullConnLayerContext();
}

void FullConnLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	size_t total = gx_dims_flatten_size( ctx->getInMD().second );

	size_t inSize = mWeights[ 0 ].size();
	size_t sampleCount = total / inSize;

	DataMatrix & gradients = ctx->getGradients();

	if( gradients.size() <= 0 ) {
		gradients.reserve( mWeights.size() );
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			gradients.emplace_back( DataVector( inSize ) );
		}
	}

	const DataType * input = std::begin( ctx->getInMD().first );
	const DataType * delta = std::begin( ctx->getDeltaMD().first );

	FullConnLayerContext * ctxImpl = dynamic_cast< FullConnLayerContext * >( ctx );
	DataVector & tempGradients = ctxImpl->getTempGradients();
	tempGradients.resize( inSize );

	for( size_t n = 0; n < sampleCount; n++, input += inSize, delta += mWeights.size() ) {
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			gx_vs_product( input,  delta[ i ], std::begin( tempGradients ), inSize );

			if( n == 0 ) {
				std::copy( std::begin( tempGradients ), std::end( tempGradients ), std::begin( gradients[ i ] ) );
			} else {
				std::transform( std::begin( tempGradients ), std::end( tempGradients ),
						std::begin( gradients[ i ] ), std::begin( gradients[ i ] ),
						std::plus< DataType >() );
			}
		}
	}
}

void FullConnLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	for( size_t n = 0; n < mWeights.size(); n++ ) {
		optim->update( &( mWeights[ n ] ), ctx.getGradients()[ n ], trainingCount, miniBatchCount );
	}

	if( !gx_is_inner_debug ) optim->updateBiases( &mBiases, ctx.getDeltaMD().first, miniBatchCount );
}

////////////////////////////////////////////////////////////

ConvLayer :: ConvLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize )
	: BaseLayer( BaseLayer::eConv )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = {
		filterCount,
		mBaseInDims[ 1 ] - filterSize + 1,
		mBaseInDims[ 2 ] - filterSize + 1
	};

	mFilters.second = { filterCount, mBaseInDims[ 0 ], filterSize, filterSize };

	mFilters.first.resize( gx_dims_flatten_size( mFilters.second) );
	for( auto & item : mFilters.first ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();

	mBiases.resize( filterCount );
	for( auto & item : mBiases ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();
}

ConvLayer :: ConvLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases )
	: BaseLayer( BaseLayer::eConv )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = {
		filters.second[ 0 ],
		mBaseInDims[ 1 ] - filters.second[ 2 ] + 1,
		mBaseInDims[ 2 ] - filters.second[ 3 ] + 1
	};

	mFilters = filters;

	mBiases = biases;
}

ConvLayer :: ~ConvLayer()
{
}

BaseLayerContext * ConvLayer :: newCtx() const
{
	ConvLayerContext * ctx = new ConvLayerContext();

	return ctx;
}

void ConvLayer :: printWeights( bool isDetail ) const
{
	printf( "\nfilterDims = %s\n", gx_vector2string( mFilters.second ).c_str() );

	if( !isDetail ) return;

	Utils::printMDVector( "filters", mFilters );
	Utils::printVector( "biases", mBiases );
}

const MDVector & ConvLayer :: getFilters() const
{
	return mFilters;
}

const DataVector & ConvLayer :: getBiases() const
{
	return mBiases;
}

void ConvLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInMD().second;

	assert( inDims.size() == 4 );

	Dims & outDims = ctx->getOutMD().second;
	outDims = {
		inDims[ 0 ], mFilters.second[ 0 ],
		inDims[ 2 ] - mFilters.second[ 2 ] + 1,
		inDims[ 3 ] - mFilters.second[ 3 ] + 1
	};
	ctx->getOutMD().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outMS( ctx->getOutMD() );

	MDSpanRO filterMS( mFilters );

	MDSpanRO inMS( ctx->getInMD() );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < filterMS.dim( 0 ); f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = forwardConv( inMS, n, f, x, y, filterMS ) + mBiases[ f ];
				}
			}
		}
	}
}

DataType ConvLayer :: forwardConv( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t c = 0; c < filterMS.dim( 1 ); c++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( sampleIndex, c, beginX + x, beginY + y ) * filterMS( filterIndex, c, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	ConvLayerContext * ctxImpl = dynamic_cast< ConvLayerContext * >( ctx );

	const Dims & outDims = ctx->getOutMD().second;

	// 1. prepare outDelta padding data
	MDVector & paddingDeltaMD = ctxImpl->getPaddingDeltaMD();
	if( paddingDeltaMD.second.size() <= 0 ) {
		paddingDeltaMD.second = {
				outDims[ 0 ],
				outDims[ 1 ],
				outDims[ 2 ] + 2 * ( mFilters.second[ 2 ] - 1 ),
				outDims[ 3 ] + 2 * ( mFilters.second[ 3 ] - 1 )
		};
	}
	paddingDeltaMD.second[ 0 ] = outDims[ 0 ];

	paddingDeltaMD.first.resize( gx_dims_flatten_size( paddingDeltaMD.second ) );

	MDSpanRW paddingDeltaMS( paddingDeltaMD );

	MDSpanRO deltaRO( ctx->getDeltaMD() );
	copyOutDelta( deltaRO, mFilters.second[ 2 ], &paddingDeltaMS );

	if( gx_is_inner_debug ) Utils::printMDVector( "paddingDelta", paddingDeltaMD );

	// 2. prepare rotate180 filters
	DataVector rot180Filters( mFilters.first.size() );
	Im2Rows::rot180Filters( mFilters.first, mFilters.second, &rot180Filters );
	if( gx_is_inner_debug ) Utils::printVector( "rot180Filters", rot180Filters, mFilters.second );

	// 3. convolution
	MDSpanRO rot180FiltersMS( rot180Filters, mFilters.second );
	MDSpanRO paddingDeltaRO( paddingDeltaMD );

	const Dims & inDims = ctx->getInMD().second;

	MDSpanRW inDeltaMS( *inDelta, inDims );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {
		for( size_t c = 0; c < inDims[ 1 ]; c++ ) {
			for( size_t x = 0; x < inDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < inDims[ 3 ]; y++ ) {
					inDeltaMS( n, c, x, y ) = backwardConv( paddingDeltaRO, n, c, x, y, rot180FiltersMS );
				}
			}
		}
	}
}

DataType ConvLayer :: backwardConv( const MDSpanRO & inMS, size_t sampleIndex, size_t channelIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t f = 0; f < filterMS.dim( 0 ); f++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( sampleIndex, f, beginX + x, beginY + y ) * filterMS( f, channelIndex, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: copyOutDelta( const MDSpanRO & outDeltaMS, size_t filterSize, MDSpanRW * outPaddingMS )
{
	for( size_t n = 0; n < outDeltaMS.dim( 0 ); n++ ) {
		for( size_t f = 0; f < outDeltaMS.dim( 1 ); f++ ) {
			for( size_t x = 0; x < outDeltaMS.dim( 2 ); x++ ) {
				for( size_t y = 0; y < outDeltaMS.dim( 3 ); y++ ) {
					( *outPaddingMS )( n, f, x + filterSize - 1, y + filterSize - 1 ) = outDeltaMS( n, f, x, y );
				}
			}
		}
	}
}

void ConvLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	const Dims & outDims = ctx->getOutMD().second;

	if( ctx->getGradients().size() <= 0 ) {
		ctx->getGradients().emplace_back( DataVector( gx_dims_flatten_size( mFilters.second ) ) );
	} else {
		ctx->getGradients()[ 0 ] = 0.0;
	}

	MDSpanRW gradientMS( ctx->getGradients()[ 0 ], mFilters.second );
	MDSpanRO deltaRO( ctx->getDeltaMD() );

	const MDSpanRO inMS( ctx->getInMD() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < mFilters.second[ 0 ]; f++ ) {
			for( size_t c = 0; c < mFilters.second[ 1 ]; c++ ) {
				for( size_t x = 0; x < mFilters.second[ 2 ]; x++ ) {
					for( size_t y = 0; y < mFilters.second[ 3 ]; y++ ) {
						gradientMS( f, c, x, y ) += gradientConv( inMS, n, f, c, x, y, deltaRO );
					}
				}
			}
		}
	}
}

DataType ConvLayer :: gradientConv( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t channelIndex, size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
		for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
			total += inMS( sampleIndex, channelIndex, beginX + x, beginY + y ) * filterMS( sampleIndex, filterIndex, x, y );
		}
	}

	return total;
}

void ConvLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	optim->update( &mFilters.first, ctx.getGradients()[ 0 ], trainingCount, miniBatchCount );

	const Dims & deltaDims = ctx.getDeltaMD().second;

	DataVector biasDelta( deltaDims[ 1 ] );

	const MDSpanRO deltaMS( ctx.getDeltaMD() );

	for( size_t n = 0; n < deltaDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < deltaDims[ 1 ]; f++ ) {
			for( size_t i = 0; i < deltaDims[ 2 ]; i++ ) {
				for( size_t j = 0; j < deltaDims[ 3 ]; j++ ) {
					biasDelta[ f ] += deltaMS( n, f, i, j );
				}
			}
		}
	}

	if( gx_is_inner_debug ) Utils::printVector( "bias.delta", biasDelta );

	optim->updateBiases( &mBiases, biasDelta, miniBatchCount );
}

////////////////////////////////////////////////////////////

ConvExLayer :: ConvExLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize )
	: ConvLayer( baseInDims, filterCount, filterSize )
{
	mType = eConvEx;

	updateFiltersRows( mFilters, &mRowsOfFilters, &mRowsOfRot180Filters );
}

ConvExLayer :: ConvExLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases )
	: ConvLayer( baseInDims, filters, biases )
{
	mType = eConvEx;

	updateFiltersRows( mFilters, &mRowsOfFilters, &mRowsOfRot180Filters );
}

ConvExLayer :: ~ConvExLayer()
{
}

BaseLayerContext * ConvExLayer :: newCtx() const
{
	ConvExLayerContext * ctx = new ConvExLayerContext();

	return ctx;
}

void ConvExLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const Dims & inDims = ctx->getInMD().second;

	Dims & outDims = ctx->getOutMD().second;

	outDims = mBaseOutDims;
	outDims.insert( outDims.begin(), inDims[ 0 ] );

	ctx->getOutMD().first.resize( gx_dims_flatten_size( outDims ) );

	Dims outDims4Rows = { outDims[ 0 ], outDims[ 1 ], outDims[ 2 ] * outDims[ 3 ] };
	MDSpanRW outMS4Rows( ctx->getOutMD().first, outDims4Rows );

	DataMatrix & rows4input = ctxImpl->getRows4calcOutput();

	MDSpanRO inMS( ctx->getInMD() );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {

		Im2Rows::input2Rows( inMS, n, mFilters.second, &rows4input );

		if( gx_is_inner_debug ) Utils::printMatrix( "input", rows4input );

		assert( ctx->getOutMD().first.size() == ( inDims[ 0 ] * rows4input.size() * mRowsOfFilters.size() ) );

		for( size_t i = 0; i < mRowsOfFilters.size(); i++ ) {
			for( size_t j = 0; j < rows4input.size(); j++ ) {
				outMS4Rows( n, i, j ) = gx_inner_product( std::begin( mRowsOfFilters[ i ] ),
						std::begin( rows4input[ j ] ), mRowsOfFilters[ i ].size() );
				outMS4Rows( n, i, j ) += mBiases[ i ];
			}
		}
	}
}

void ConvExLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	const Dims & outDims = ctx->getOutMD().second;

	MDVector & paddingDeltaMD = ctxImpl->getPaddingDeltaMD();

	// prepare outDelta padding data
	if( paddingDeltaMD.second.size() <= 0 ) {
		paddingDeltaMD.second = {
				outDims[ 0 ],
				outDims[ 1 ],
				outDims[ 2 ] + 2 * ( mFilters.second[ 2 ] - 1 ),
				outDims[ 3 ] + 2 * ( mFilters.second[ 3 ] - 1 )
		};
	}
	paddingDeltaMD.second[ 0 ] = outDims[ 0 ];

	paddingDeltaMD.first.resize( gx_dims_flatten_size( paddingDeltaMD.second ) );

	MDSpanRW paddingDeltaMS( paddingDeltaMD );
	MDSpanRO deltaRO( ctx->getDeltaMD() );
	copyOutDelta( deltaRO, mFilters.second[ 2 ], &paddingDeltaMS );

	if( gx_is_inner_debug ) Utils::printMDVector( "outPadding", paddingDeltaMD );

	if( gx_is_inner_debug ) Utils::printMatrix( "rot180filters", mRowsOfRot180Filters );

	MDSpanRO paddingDeltaRO( paddingDeltaMD );

	Dims inDeltaDims = { outDims[ 0 ], mBaseInDims[ 0 ], gx_dims_flatten_size( mBaseInDims ) / mBaseInDims[ 0 ] };
	MDSpanRW inDeltaMS( *inDelta, inDeltaDims );

	// rot180Filters dims
	Dims fakeDims = { mFilters.second[ 1 ], mFilters.second[ 0 ], mFilters.second[ 2 ], mFilters.second[ 3 ] };

	DataMatrix & rows4delta = ctxImpl->getRows4backpropagate();

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		Im2Rows::input2Rows( paddingDeltaRO, n, fakeDims, &rows4delta );

		//if( gx_is_inner_debug ) Utils::printMatrix( "outPadding", rows4delta );

		for( size_t i = 0; i < mRowsOfRot180Filters.size(); i++ ) {
			for( size_t j = 0; j < rows4delta.size(); j++ ) {
				inDeltaMS( n, i, j ) = gx_inner_product( std::begin( mRowsOfRot180Filters[ i ] ),
						std::begin ( rows4delta[ j ] ), mRowsOfRot180Filters[ i ].size() );
			}
		}
	}
}

void ConvExLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	if( ctx->getGradients().size() <= 0 ) {
		ctx->getGradients().emplace_back( DataVector( gx_dims_flatten_size( mFilters.second ) ) );
	} else {
		ctx->getGradients()[ 0 ] = 0.0;
	}

	const MDSpanRO inMS( ctx->getInMD() );

	const Dims & outDims = ctx->getOutMD().second;

	Dims gradientDims = { mFilters.second[ 0 ], gx_dims_flatten_size( mFilters.second ) / mFilters.second[ 0 ] };

	MDSpanRW gradientMS( ctx->getGradients()[ 0 ], gradientDims );

	DataMatrix & rowsOfDelta = ctxImpl->getRowsOfDelta();
	DataMatrix & rows4input = ctxImpl->getRows4collectGradients();

	for( size_t n = 0; n < inMS.dim( 0 ); n++ ) {

		Im2Rows::deltas2Rows( ctx->getDeltaMD().first, n, outDims, &rowsOfDelta );

		if( gx_is_inner_debug ) Utils::printMatrix( "deltas", rowsOfDelta );

		Im2Rows::input2Rows4Gradients( inMS, n, outDims, &rows4input );

		if( gx_is_inner_debug ) Utils::printMatrix( "input", rows4input );

		for( size_t i = 0; i < rowsOfDelta.size(); i++ ) {
			for( size_t j = 0; j < rows4input.size(); j++ ) {
				gradientMS( i, j ) += gx_inner_product( std::begin( rowsOfDelta[ i ] ),
						std::begin( rows4input[ j ] ), rowsOfDelta[ i ].size() );
			}
		}
	}
}

void ConvExLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	ConvLayer::applyGradients( ctx, optim, trainingCount, miniBatchCount );

	updateFiltersRows( mFilters, &mRowsOfFilters, &mRowsOfRot180Filters );
}


void ConvExLayer :: updateFiltersRows( const MDVector & filters,
		DataMatrix * rowsOfFilters, DataMatrix * rowsOfRot180Filters )
{
	Im2Rows::filters2Rows( filters.first, filters.second, rowsOfFilters );
	Im2Rows::rot180Filters2Rows( filters.first, filters.second, rowsOfRot180Filters );

	if( gx_is_inner_debug ) Utils::printMatrix( "filters", *rowsOfFilters );
	if( gx_is_inner_debug ) Utils::printMatrix( "rot180filters", *rowsOfRot180Filters );
}

////////////////////////////////////////////////////////////

MaxPoolLayer :: MaxPoolLayer( const Dims & baseInDims, size_t poolSize )
	: BaseLayer( BaseLayer::eMaxPool )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { mBaseInDims[ 0 ], mBaseInDims[ 1 ] / poolSize, mBaseInDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

MaxPoolLayer :: ~MaxPoolLayer()
{
}

void MaxPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t MaxPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

BaseLayerContext * MaxPoolLayer :: newCtx() const
{
	BaseLayerContext * ctx = new BaseLayerContext();

	return ctx;
}

void MaxPoolLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInMD().second;

	Dims & outDims = ctx->getOutMD().second;

	outDims = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	ctx->getOutMD().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outMS( ctx->getOutMD() );
	MDSpanRO inMS( ctx->getInMD() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = pool( inMS, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType MaxPoolLayer :: pool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 1.0 * INT_MIN;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result = std::max( result, inMS( sampleIndex, filterIndex, beginX + x, beginY + y ) );
		}
	}

	return result;
}

void MaxPoolLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	const Dims & inDims = ctx->getInMD().second;
	const Dims & outDims = ctx->getOutMD().second;

	inDelta->resize( ctx->getInMD().first.size() );

	MDSpanRW inDeltaMS( *inDelta, inDims );

	MDSpanRW outMS( ctx->getOutMD() );

	MDSpanRO inMS( ctx->getInMD() );
	MDSpanRO outDeltaMS( ctx->getDeltaMD() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inMS, n, f, x * mPoolSize, y * mPoolSize,
							outMS( n, f, x, y ), outDeltaMS( n, f, x, y ), &inDeltaMS );
				}
			}
		}
	}
}

void MaxPoolLayer :: unpool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY,
		const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			if( inMS( sampleIndex, filterIndex, beginX + x, beginY + y ) == maxValue ) {
				( *inDeltaMS )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta;
			}
		}
	}
}

////////////////////////////////////////////////////////////

AvgPoolLayer :: AvgPoolLayer( const Dims & baseInDims, size_t poolSize )
	: BaseLayer( BaseLayer::eAvgPool )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { mBaseInDims[ 0 ], mBaseInDims[ 1 ] / poolSize, mBaseInDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

AvgPoolLayer :: ~AvgPoolLayer()
{
}

void AvgPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t AvgPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

BaseLayerContext * AvgPoolLayer :: newCtx() const
{
	BaseLayerContext * ctx = new BaseLayerContext();

	return ctx;
}

void AvgPoolLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInMD().second;

	Dims & outDims = ctx->getOutMD().second;

	outDims = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	ctx->getOutMD().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outMS( ctx->getOutMD() );
	MDSpanRO inMS( ctx->getInMD() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = pool( inMS, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType AvgPoolLayer :: pool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 0;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result += inMS( sampleIndex, filterIndex, beginX + x, beginY + y );
		}
	}

	return result / ( mPoolSize * mPoolSize );
}

void AvgPoolLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	const Dims & inDims = ctx->getInMD().second;
	const Dims & outDims = ctx->getOutMD().second;

	inDelta->resize( ctx->getInMD().first.size() );

	MDSpanRW inDeltaMS( *inDelta, inDims );

	MDSpanRW outMS( ctx->getOutMD() );

	MDSpanRO inMS( ctx->getInMD() );
	MDSpanRO outDeltaMS( ctx->getDeltaMD() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inMS, n, f, x * mPoolSize, y * mPoolSize,
							outMS( n, f, x, y ), outDeltaMS( n, f, x, y ), &inDeltaMS );
				}
			}
		}
	}
}

void AvgPoolLayer :: unpool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			( *inDeltaMS )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta / ( mPoolSize * mPoolSize );
		}
	}
}

////////////////////////////////////////////////////////////

DropoutLayer :: DropoutLayer( const Dims & baseInDims, DataType dropRate )
	: BaseLayer( eDropout )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = baseInDims;

	mDropRate = dropRate;
	mIsTraining = false;
}

DropoutLayer :: ~DropoutLayer()
{
}

void DropoutLayer :: printWeights( bool isDetail ) const
{
	printf( "\nDropRate = %f\n", mDropRate );
}

DataType DropoutLayer :: getDropRate() const
{
	return mDropRate;
}

BaseLayerContext * DropoutLayer :: newCtx() const
{
	DropoutLayerContext * ctx = new DropoutLayerContext();

	return ctx;
}

void DropoutLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	DropoutLayerContext * ctxImpl = dynamic_cast< DropoutLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const DataVector & input = ctx->getInMD().first;
	DataVector & output = ctx->getOutMD().first;

	ctx->getOutMD().second = ctx->getInMD().second;
	ctx->getOutMD().first.resize( input.size() );

	BoolVector & mask = ctxImpl->getMask();
	mask.resize( input.size() );

	if( mIsTraining ) {
		for( size_t i = 0; i < input.size(); i++ ) {
			if( Utils::random( 0, 1 ) < mDropRate ) {
				mask[ i ] = true;
				output[ i ] = 0;
			} else {
				mask[ i ] = false;
				output[ i ] = input[ i ] / ( 1.0 - mDropRate );
			}
		}
	} else {
		output = input;
	}

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "dropout.input", ctx->getInMD() );
		Utils::printMDVector( "dropout.output", ctx->getOutMD() );
	}
}

void DropoutLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	DropoutLayerContext * ctxImpl = dynamic_cast< DropoutLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const DataVector & delta = ctx->getDeltaMD().first;
	BoolVector & mask = ctxImpl->getMask();

	for( size_t i = 0; i < delta.size(); i++ )
			( *inDelta )[ i ] = mask[ i ] ? 0 : delta[ i ];

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "dropout.outDelta", ctx->getDeltaMD() );
		Utils::printVector( "dropout.inDelta", *inDelta );
	}
}


}; // namespace gxnet;

