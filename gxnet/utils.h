#pragma once

#include "common.h"
#include "context.h"

#include <algorithm>

namespace gxnet {

typedef struct tagCmdArgs {
	int mThreadCount;
	int mTrainingCount;
	int mEvalCount;
	int mEpochCount;
	int mMiniBatchCount;
	DataType mLearningRate;
	DataType mLambda;
	bool mIsShuffle;
	const char * mModelPath;
} CmdArgs_t;

class Network;

class Utils {
public:
	template< class ForwardIt >
	static size_t max_index( ForwardIt first, ForwardIt last )
	{
		return std::distance( first, std::max_element( first, last ) );
	}

	static DataType random();

	static DataType random( DataType min, DataType max );

	static DataType calcSSE( const DataVector & output, const DataVector & target );

	static void printMnistImage( const char * tag, const DataVector & data );

	static bool centerMnistImage( DataVector & orgImage, DataVector * newImage );

	static bool expandMnistImage( DataVector & orgImage, DataVector * newImage );

	static bool loadMnistImages( const int limitCount, const char * path, DataMatrix * images );

	static bool loadMnistLabels( int limitCount, const char * path, DataMatrix * labels );

	static void printMatrix( const char * tag, const DataMatrix & data,
			bool useSciFmt = true, bool colorMax = false);

	static void printCtx( const char * tag, const BaseLayerCtxPtrVector & data );

	static void printCtx( const char * tag, const BackwardCtxPtrVector & data );

	static void printVector( const char * tag, const DataVector & data, bool useSciFmt = true );

	static void printVector( const char * tag, const DataVector & data, const Dims & dims, bool useSciFmt = true );

	static void printMDSpan( const char * tag, const MDSpanRO & data, bool useSciFmt = true );

	static bool save( const char * path, const Network & network );

	static bool load( const char * path, Network * network );

public:

	static void getCmdArgs( int argc, char * const argv[],
			const CmdArgs_t & defaultArgs, CmdArgs_t * args );
};

}; // namespace gxnet;

