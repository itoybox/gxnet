#pragma once

#include "common.h"
#include "layer.h"
#include "context.h"
#include "optim.h"
#include "utils.h"

#include <tuple>

namespace gxnet {

class Network;

typedef void ( * OnEpochEnd_t )( Network & network, int epoch, DataType loss );

typedef std::tuple<
			const DataMatrix * /* input */,
			const DataMatrix * /* target */
		> TrainingData;

typedef std::tuple<
			const IntVector * /* idxOfData */,
			size_t            /* chunkBegin */,
			size_t            /* chunkEnd */
		> ChunkInfo;

class NetworkContext {
public:
	NetworkContext();
	~NetworkContext();

	// training data related methods
	void setTrainingData( const TrainingData & data );

	const TrainingData & getTrainingData();

	// chunk related methods
	void setChunkInfo( const ChunkInfo & info );

	const ChunkInfo & getChunkInfo();

	// layer context
	BaseLayerContext * getLayerCtx( size_t index );

	BaseLayerContextPtrVector & getLayerCtx();

	// mini batch workspace
	MDVector & getInput();

	MDVector & getTarget();

	// mini batch backward context
	BackwardContext * getBatchBwdCtx( size_t index );

	BackwardContextPtrVector & getBatchBwdCtx();

	void clearBatch();

	void addToBatch();

private:
	BaseLayerContextPtrVector mLayerCtx;
	BackwardContextPtrVector mBatchBwdCtx;

	TrainingData mTrainingData;
	ChunkInfo mChunkInfo;

	MDVector mInput, mTarget;
};

class Network {
public:
	enum { eMeanSquaredError = 1, eCrossEntropy = 2 };

	Network( int lossFuncType = eMeanSquaredError );
	~Network();

	void setOnEpochEnd( OnEpochEnd_t onEpochEnd );

	void setTraining( bool isTraining );

	void setLossFuncType( int lossFuncType );

	int getLossFuncType() const;

	void addLayer( BaseLayer * layer );

	BaseLayerPtrVector & getLayers();

	const BaseLayerPtrVector & getLayers() const;

	void initCtx( NetworkContext * ctx ) const;

	bool forward( const DataVector & input, DataVector * output ) const;

	bool forward( const DataMatrix & input, DataMatrix * output ) const;

	bool train( const DataMatrix & input, const DataMatrix & target, const CmdArgs_t & args,
			DataVector * losses = nullptr );

	void print( bool isDetail = false ) const;

	bool trainMiniBatch( NetworkContext * ctx, DataType * totalLoss );

private:

	bool forward( NetworkContext * ctx ) const;

	bool backward( NetworkContext * ctx, const MDVector & targetMD ) const;

	void collect( NetworkContext * ctx ) const;

	bool apply( NetworkContext * ctx, Optim * optim, size_t trainingCount, size_t miniBatchCount );

	DataType calcLoss( const DataVector & target, const DataVector & output );

	bool trainInternal( const DataMatrix & input, const DataMatrix & target, const CmdArgs_t & args,
			DataVector * losses = nullptr );

private:
	OnEpochEnd_t mOnEpochEnd;
	int mLossFuncType;
	BaseLayerPtrVector mLayers;
	bool mIsTraining;
};

}; // namespace gxnet;

