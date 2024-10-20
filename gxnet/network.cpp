
#include "network.h"
#include "utils.h"

#include <random>
#include <numeric>
#include <algorithm>
#include <memory>

#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>

namespace gxnet {

NetworkContext :: NetworkContext()
{
}

NetworkContext :: ~NetworkContext()
{
	for( auto & item : mLayerCtx ) delete item;
	for( auto & item : mBatchBwdCtx ) delete item;
}

void NetworkContext :: setTrainingData( const TrainingData & data )
{
	mTrainingData = data;
}

const TrainingData & NetworkContext :: getTrainingData()
{
	return mTrainingData;
}

void NetworkContext :: setChunkInfo( const ChunkInfo & info )
{
	mChunkInfo = info;
}

const ChunkInfo & NetworkContext :: getChunkInfo()
{
	return mChunkInfo;
}

MDVector & NetworkContext :: getInputMD()
{
	return mInputMD;
}

MDVector & NetworkContext :: getTargetMD()
{
	return mTargetMD;
}

BaseLayerContext * NetworkContext :: getLayerCtx( size_t index )
{
	return mLayerCtx[ index ];
}

BaseLayerContextPtrVector & NetworkContext :: getLayerCtx()
{
	return mLayerCtx;
}

BackwardContext * NetworkContext :: getBatchBwdCtx( size_t index )
{
	return mBatchBwdCtx[ index ];
}

BackwardContextPtrVector & NetworkContext :: getBatchBwdCtx()
{
	return mBatchBwdCtx;
}

void NetworkContext :: clearBatch()
{
	for( auto & ctx : mBatchBwdCtx ) {
		for( auto & vec : ctx->getGradients() ) vec = 0.0;
		ctx->getDeltaMD().first = 0.0;
	}
}

void NetworkContext :: addToBatch()
{
	if( mBatchBwdCtx.size() <= 0 ) {
		for( size_t i = 0; i < mLayerCtx.size(); i++ ) {
			mBatchBwdCtx.emplace_back( new BackwardContext() );

			Dims outDims = mLayerCtx[ i ]->getOutMD().second;
			outDims[ 0 ] = 1;
			mBatchBwdCtx[ i ]->getDeltaMD().second = outDims;
			mBatchBwdCtx[ i ]->getDeltaMD().first.resize( gx_dims_flatten_size( outDims ) );

			for( auto & item : mLayerCtx[ i ]->getGradients() ) {
				mBatchBwdCtx[ i ]->getGradients().emplace_back( DataVector( item.size() ) );
			}
		}
	}

	for( size_t i = 0; i < mBatchBwdCtx.size(); i++ ) {
		DataVector & delta = mBatchBwdCtx[ i ]->getDeltaMD().first;

		size_t total = mLayerCtx[ i ]->getDeltaMD().first.size();
		const DataType * deltaPtr = std::begin( mLayerCtx[ i ]->getDeltaMD().first );
		for( size_t index = 0; index < total; index += delta.size(), deltaPtr += delta.size() ) {
			std::transform( deltaPtr, deltaPtr + delta.size(),
					std::begin( delta ), std::begin( delta ), std::plus<DataType>() );
		}

		for( size_t j = 0; j < mBatchBwdCtx[ i ]->getGradients().size(); j++ ) {
			//mBatchBwdCtx[ i ]->getGradients()[ j ] += mLayerCtx[ i ]->getGradients()[ j ];

			std::transform( std::begin( mLayerCtx[ i ]->getGradients()[ j ] ),
					std::end( mLayerCtx[ i ]->getGradients()[ j ] ),
					std::begin( mBatchBwdCtx[ i ]->getGradients()[ j ] ),
					std::begin( mBatchBwdCtx[ i ]->getGradients()[ j ] ),
					std::plus< DataType >() );
		}
	}
}

////////////////////////////////////////////////////////////

Network :: Network( int lossFuncType )
{
	mOnEpochEnd = NULL;
	mLossFuncType = lossFuncType;
	mIsTraining = false;
}

Network :: ~Network()
{
	for( auto & item : mLayers ) delete item;
}

void Network :: print( bool isDetail ) const
{
	printf( "\n{{{ isDetail %s\n", isDetail ? "true" : "false" );
	printf( "Network: LayerCount = %zu; LossFuncType = %d;\n", mLayers.size(), mLossFuncType );
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		printf( "\nLayer#%ld: ", i );
		layer->print( isDetail );
	}
	printf( "}}}\n\n" );
}

void Network :: setOnEpochEnd( OnEpochEnd_t onEpochEnd )
{
	mOnEpochEnd = onEpochEnd;
}

void Network :: setTraining( bool isTraining )
{
	mIsTraining = isTraining;

	for( auto & layer : mLayers ) layer->setTraining( isTraining );
}

void Network :: setLossFuncType( int lossFuncType )
{
	mLossFuncType = lossFuncType;
}

int Network :: getLossFuncType() const
{
	return mLossFuncType;
}

BaseLayerPtrVector & Network :: getLayers()
{
	return mLayers;
}

const BaseLayerPtrVector & Network :: getLayers() const
{
	return mLayers;
}

void Network :: addLayer( BaseLayer * layer )
{
	layer->setTraining( mIsTraining );

	mLayers.emplace_back( layer );
}

bool Network :: forward( const DataVector & input, DataVector * output ) const
{
	NetworkContext ctx;
	initCtx( &ctx );

	Dims dims = mLayers[ 0 ]->getBaseInDims();
	dims.insert( dims.begin(), 1 );

	MDVector inMD( input, dims );

	ctx.getLayerCtx( 0 )->setInMD( &inMD );

	bool ret = forward( &ctx );

	*output = ctx.getLayerCtx().back()->getOutMD().first;

	return ret;
}

bool Network :: forward( const DataMatrix & input, DataMatrix * output ) const
{
	NetworkContext ctx;
	initCtx( &ctx );

	Dims dims = mLayers[ 0 ]->getBaseInDims();
	dims.insert( dims.begin(), 1 );

	output->reserve( input.size() );

	bool ret = true;

	for( const auto & item : input ) {

		MDVector inMD( item, dims );

		ctx.getLayerCtx( 0 )->setInMD( &inMD );

		ret = forward( &ctx );

		if( !ret ) break;

		output->push_back( ctx.getLayerCtx().back()->getOutMD().first );
	}

	return ret;
}

bool Network :: forward( NetworkContext * ctx ) const
{
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		BaseLayerContext * layerCtx = ctx->getLayerCtx( i );

		layer->forward( layerCtx );
	}

	return true;
}

bool Network :: apply( NetworkContext * ctx, Optim * optim,
		size_t trainingCount, size_t miniBatchCount )
{
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];
		layer->applyGradients( *( ctx->getBatchBwdCtx( i ) ), optim, trainingCount, miniBatchCount );
	}

	return true;
}

void Network :: collect( NetworkContext * ctx ) const
{
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		layer->collectGradients( ctx->getLayerCtx( i ) );
	}

	if( gx_is_inner_debug ) Utils::printCtx( "collect", ctx->getLayerCtx() );
}

bool Network :: backward( NetworkContext * ctx, const MDVector & targetMD ) const
{
	BaseLayerContext * layerCtx = ctx->getLayerCtx().back();

	const DataVector & lastOutput = layerCtx->getOutMD().first;

	const DataVector & target = targetMD.first;

	assert( layerCtx->getDeltaMD().first.size() == target.size() );

	if( eMeanSquaredError == mLossFuncType ) {
		layerCtx->getDeltaMD().first = 2.0 * ( lastOutput - target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		layerCtx->getDeltaMD().first = lastOutput - target;
	}

	for( ssize_t i = mLayers.size() - 1; i >= 0; i-- ) {
		DataVector * inDelta = ( i > 0 ) ? &( ctx->getLayerCtx()[ i - 1 ]->getDeltaMD().first ) : NULL;

		BaseLayer * layer = mLayers[ i  ];

		layer->backward( ctx->getLayerCtx()[ i ], inDelta );
	}

	return true;
}

void Network :: initCtx( NetworkContext * ctx ) const
{
	BaseLayerContext * layerCtx = NULL;

	for( auto & layer : mLayers ) {
		layerCtx = layer->createCtx();
		ctx->getLayerCtx().push_back( layerCtx );
	}

	for( size_t i = 1; i < mLayers.size(); i++ ) {
		BaseLayerContext * prev = ctx->getLayerCtx( i - 1 );
		BaseLayerContext * curr = ctx->getLayerCtx( i );

		curr->setInMD( &( prev->getOutMD() ) );
	}
}

DataType Network :: calcLoss( const DataVector & target, const DataVector & output )
{
	DataType ret = 0;

	if( eMeanSquaredError == mLossFuncType ) {
		ret = Utils::calcSSE( output, target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		for( size_t x = 0; x < target.size(); x++ ) {
			DataType y = target[ x ], a = output[ x ];
			DataType tmp = y * std::log( a ); // + ( 1 - y ) * std::log( 1 - a );
			ret -= tmp;
		}
	}

	return ret;
}

bool Network :: trainMiniBatch( NetworkContext * ctx, DataType * totalLoss )
{
	const DataMatrix * input = std::get<0>( ctx->getTrainingData() );
	const DataMatrix * target = std::get<1>( ctx->getTrainingData() );

	const IntVector * idxOfData = std::get<0>( ctx->getChunkInfo() );
	size_t chunkBegin = std::get<1>( ctx->getChunkInfo() );
	size_t chunkEnd = std::get<2>( ctx->getChunkInfo() );

	MDVector & inputMD = ctx->getInputMD();
	if( inputMD.second.size() <= 0 ) {
		inputMD.second = mLayers[ 0 ]->getBaseInDims();
		inputMD.second.insert( inputMD.second.begin(), 1 );
	}
	inputMD.second[ 0 ] = chunkEnd - chunkBegin;
	inputMD.first.resize( gx_dims_flatten_size( inputMD.second ) );

	MDVector & targetMD = ctx->getTargetMD();
	if( targetMD.second.size() <= 0 ) {
		targetMD.second = { 1, ( *target )[ 0 ].size() };
	}
	targetMD.second[ 0 ] = chunkEnd - chunkBegin;
	targetMD.first.resize( gx_dims_flatten_size( targetMD.second ) );

	DataType * inPtr = std::begin( inputMD.first );
	DataType * targetPtr = std::begin( targetMD.first );

	for( size_t i = chunkBegin; i < chunkEnd; i++ ) {
		const DataVector & currInput = ( *input )[ ( *idxOfData )[ i ] ];
		const DataVector & currTarget = ( *target )[ ( *idxOfData )[ i ] ];

		std::copy( std::begin( currInput ), std::end( currInput ), inPtr );
		inPtr += currInput.size();

		std::copy( std::begin( currTarget ), std::end( currTarget ), targetPtr );
		targetPtr += currTarget.size();
	}

	ctx->getLayerCtx( 0 )->setInMD( &inputMD );

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "input", inputMD );
		Utils::printMDVector( "target", targetMD );
	}

	forward( ctx );

	backward( ctx, targetMD );

	collect( ctx );

	ctx->addToBatch();

	DataType loss = calcLoss( targetMD.first, ctx->getLayerCtx().back()->getOutMD().first );

	*totalLoss += loss;

	loss /= chunkEnd - chunkBegin;

	if( gx_is_inner_debug ) {
		printf( "DEBUG: input #%ld loss %.8f totalLoss %.8f\n", chunkBegin, loss, *totalLoss );
	}

	return true;
}

bool Network :: trainInternal( const DataMatrix & input, const DataMatrix & target,
		const CmdArgs_t & args, DataVector * losses )
{
	if( input.size() != target.size() ) return false;

	setTraining( true );

	time_t beginTime = time( NULL );

	printf( "%s\tstart train, input { %zu }, target { %zu }\n",
			ctime( &beginTime ), input.size(), target.size() );

	int logInterval = args.mEpochCount / 10;
	int progressInterval = ( input.size() / args.mMiniBatchCount ) / 10;

	std::random_device rd;
	std::mt19937 gen( rd() );

	NetworkContext ctx;
	initCtx( &ctx );
	ctx.setTrainingData( TrainingData( &input, &target ) );

	std::unique_ptr< Optim > optim( Optim::SGD( args.mLearningRate, args.mLambda ) );

	if( NULL != losses ) losses->resize( args.mEpochCount, 0 );

	for( int n = 0; n < args.mEpochCount; n++ ) {

		IntVector idxOfData( input.size() );
		std::iota( idxOfData.begin(), idxOfData.end(), 0 );
		if( args.mIsShuffle ) std::shuffle( idxOfData.begin(), idxOfData.end(), gen );

		DataType totalLoss = 0;

		int miniBatchCount = std::max( args.mMiniBatchCount, 1 );

		for( size_t begin = 0; begin < idxOfData.size(); ) {
			size_t end = std::min( idxOfData.size(), begin + miniBatchCount );

			ctx.clearBatch();

			ctx.setChunkInfo( ChunkInfo( &idxOfData, begin, end ) );

			trainMiniBatch( &ctx, &totalLoss );

			if( gx_is_inner_debug ) Utils::printCtx( "batch", ctx.getBatchBwdCtx() );

			apply( &ctx, optim.get(), input.size(), end - begin );

			if( gx_is_inner_debug ) print( true );

			if( progressInterval > 0 && 0 == ( begin % ( progressInterval * miniBatchCount ) ) ) {
				printf( "\33[2K\r%zu / %zu, loss %.8f", end, idxOfData.size(), totalLoss / end );
				fflush( stdout );
			}

			begin += miniBatchCount;
			end = begin + miniBatchCount;
		}

		if( NULL != losses ) ( *losses )[ n ] = totalLoss / input.size();

		if( logInterval <= 1 || ( logInterval > 1 && 0 == n % logInterval ) || n == ( args.mEpochCount - 1 ) ) {
			time_t currTime = time( NULL );
			printf( "\33[2K\r%s\tinterval %ld [>] epoch %d, lr %f, loss %.8f\n",
				ctime( &currTime ), currTime - beginTime, n, args.mLearningRate, totalLoss / input.size() );
			beginTime = time( NULL );
		}

		if( mOnEpochEnd ) mOnEpochEnd( *this, n, totalLoss / input.size() );
	}

	return true;
}

bool Network :: train( const DataMatrix & input, const DataMatrix & target,
		const CmdArgs_t & args, DataVector * losses )
{
	std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();	

	bool ret = trainInternal( input, target, args, losses );

	std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();	

	auto timeSpan = std::chrono::duration_cast<std::chrono::milliseconds>( endTime - beginTime );

	printf( "Elapsed time: %.3f\n", timeSpan.count() / 1000.0 );

	return ret;
}


}; // namespace gxnet;

