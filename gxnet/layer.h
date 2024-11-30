#pragma once

#include "common.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace gxnet {

class BaseLayer;
class BaseLayerContext;
class BackwardContext;

typedef std::vector< BaseLayer * > BaseLayerPtrVector;

class ActFunc;
class Optim;

class BaseLayer {
public:
	enum {
		eFullConn = 1,
		eConv = 10, eMaxPool = 11, eAvgPool = 12, eConvEx = 13,
		eDropout = 20
	};

public:
	BaseLayer( int type );

	virtual ~BaseLayer();

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

public:

	virtual void print( bool isDetail = false ) const;

	void forward( BaseLayerContext * ctx ) const;

	void backward( BaseLayerContext * ctx, MDVector * inDelta ) const;

	BaseLayerContext * createCtx() const;

protected:

	virtual void printWeights( bool isDetail ) const = 0;

	virtual BaseLayerContext * newCtx() const = 0;

	virtual void calcOutput( BaseLayerContext * ctx ) const = 0;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const = 0;

public:
	int getType() const;

	void setActFunc( ActFunc * actFunc );

	const ActFunc * getActFunc() const;

	void setTraining( bool isTraining );

	const Dims & getBaseInDims() const;

	size_t getBaseInSize() const;

	const Dims & getBaseOutDims() const;

	size_t getBaseOutSize() const;

protected:
	int mType;
	bool mIsTraining;

	ActFunc * mActFunc;

	Dims mBaseInDims, mBaseOutDims;
};

class FullConnLayer : public BaseLayer {
public:
	FullConnLayer( const Dims & baseInDims, size_t neuronCount );

	~FullConnLayer();

	// for debug
	void setWeights( const MDVector & weights, const DataVector & biases );

	const MDVector & getWeights() const;

	const DataVector & getBiases() const;

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	/**
	 * input dims : (*) always convert to (N,Hin) internal, Hin=in_features
	 * output dims : (N,Hout) Hout=out_features.
	 */
	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

private:
	MDVector mWeights;
	DataVector mBiases;
};

class ConvLayer : public BaseLayer {
public:
	ConvLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize );
	ConvLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases );

	~ConvLayer();

	const MDVector & getFilters() const;

	const DataVector & getBiases() const;

public:

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	/**
	 * input dims: (N,Cin,Hin,Win) 
	 * output dims: (N,Cout,Hout,Wout)
	 */
	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

public:

	static DataType forwardConv( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
			size_t beginX, size_t beginY, const MDSpanRO & filterRO );

	static DataType backwardConv( const MDSpanRO & inRO, size_t sampleIndex, size_t channelIndex,
			size_t beginX, size_t beginY, const MDSpanRO & filterRO );

	static DataType gradientConv( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
			size_t channelIndex, size_t beginX, size_t beginY, const MDSpanRO & filterRO );

	static void copyOutDelta( const MDSpanRO & outDeltaRO, size_t filterSize, MDSpanRW * outPaddingRW );

protected:
	MDVector mFilters;
	DataVector mBiases;
};

class ConvExLayer : public ConvLayer {
public:
	ConvExLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize );
	ConvExLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases );

	~ConvExLayer();

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

protected:

	virtual BaseLayerContext * newCtx() const;

	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

private:
	MDVector mRowsOfRot180Filters;
};

class MaxPoolLayer : public BaseLayer {
public:
	MaxPoolLayer( const Dims & baseInDims, size_t poolSize );
	~MaxPoolLayer();

	size_t getPoolSize() const;

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

private:
	DataType pool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY ) const;

	void unpool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY,
			const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaRW ) const;

private:
	size_t mPoolSize;
};

class AvgPoolLayer : public BaseLayer {
public:
	AvgPoolLayer( const Dims & baseInDims, size_t poolSize );
	~AvgPoolLayer();

	size_t getPoolSize() const;

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

private:
	DataType pool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY ) const;

	void unpool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY,
			const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaRW ) const;

private:
	size_t mPoolSize;
};

class DropoutLayer : public BaseLayer {
public:
	DropoutLayer( const Dims & baseInDims, DataType dropRate );
	~DropoutLayer();

	DataType getDropRate() const;

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const;

private:
	DataType mDropRate;
};

}; // namespace gxnet;

