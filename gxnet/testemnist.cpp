#include "network.h"
#include "activation.h"
#include "utils.h"
#include "eval.h"

#include <unistd.h>

using namespace gxnet;

bool loadData( const CmdArgs_t & args, DataMatrix * input, DataMatrix * target,
		DataMatrix * input4eval, DataMatrix * target4eval )
{
	const char * path = "emnist/train-images-idx3-ubyte";
	if( ! Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "emnist/train-labels-idx1-ubyte";
	if( ! Utils::loadMnistLabels( args.mTrainingCount, path, target ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	if( args.mIsDataAug ) {
		// load rotated images
		path = "emnist/train-images-idx3-ubyte.rot";
		if( 0 == access( path, F_OK ) ) {
			if( ! Utils::loadMnistImages( args.mTrainingCount, path, input ) ) {
				printf( "read %s fail\n", path );
				return false;
			}

			path = "emnist/train-labels-idx1-ubyte.rot";
			if( ! Utils::loadMnistLabels( args.mTrainingCount, path, target ) ) {
				printf( "read %s fail\n", path );
				return false;
			}
		}

		// center mnist images
		size_t orgSize = input->size();

		for( size_t i = 0; i < orgSize; i++ ) {
			DataVector newImage;
			if( Utils::centerMnistImage( input->at( i ), &newImage ) ) {
				input->emplace_back( newImage );
				target->emplace_back( target->at( i ) );
			}
		}

		printf( "center %zu images\n", input->size() - orgSize );
	}

	path = "emnist/test-images-idx3-ubyte";
	if( ! Utils::loadMnistImages( args.mEvalCount, path, input4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	path = "emnist/test-labels-idx1-ubyte";
	if( ! Utils::loadMnistLabels( args.mEvalCount, path, target4eval ) ) {
		printf( "read %s fail\n", path );
		return false;
	}

	printf( "input { %zu }, target { %zu }, input4eval { %zu }, target4eval { %zu }\n",
			input->size(), target->size(), input4eval->size(), target4eval->size() );

	// convert to 32 * 32
	for( auto & item : *input ) {
		DataVector orgImage = item;
		Utils::expandMnistImage( orgImage, &item );
	}

	for( auto & item : *input4eval ) {
		DataVector orgImage = item;
		Utils::expandMnistImage( orgImage, &item );
	}

	return true;
}

void save_checkpoint( Network & network, int epoch, DataType loss )
{
	char path[ 128 ] = { 0 };
	snprintf( path, sizeof( path ), "./emnist.%d.model", epoch );

	Utils::save( path, network );
	printf( "\33[2K\r\tsave checkpoint (%s) for epoch#%d, loss %f\n", path, epoch, loss );
}

void test( const CmdArgs_t & args )
{
	DataMatrix input, target, input4eval, target4eval;

	if( ! loadData( args, &input, &target, &input4eval, &target4eval ) ) {
		printf( "loadData fail\n" );
		return;
	}

	const char * path = "./emnist.model";

	Dims baseInDims = { 1, (size_t)std::sqrt( input[ 0 ].size() ), (size_t)std::sqrt( input[ 0 ].size() ) };

	//train & save model
	{
		Network network;

		network.setOnEpochEnd( save_checkpoint );
		network.setLossFuncType( Network::eCrossEntropy );

		if( NULL != args.mModelPath && 0 == access( args.mModelPath, F_OK ) ) {
			if(  Utils::load( args.mModelPath, &network ) ) {
				printf( "continue training %s\n", args.mModelPath );
			} else {
				printf( "load( %s ) fail\n", args.mModelPath );
				return;
			}
		} else {
			BaseLayer * layer = NULL;

			layer = new ConvExLayer( baseInDims, 4, 5 );
			layer->setActFunc( ActFunc::leakyReLU() );
			network.addLayer( layer );

			layer = new MaxPoolLayer( layer->getBaseOutDims(), 2 );
			network.addLayer( layer );

			layer = new ConvExLayer( layer->getBaseOutDims(), 8, 3 );
			layer->setActFunc( ActFunc::leakyReLU() );
			network.addLayer( layer );

			layer = new MaxPoolLayer( layer->getBaseOutDims(), 2 );
			network.addLayer( layer );

			layer = new FullConnLayer( layer->getBaseOutDims(), 60 );
			layer->setActFunc( ActFunc::sigmoid() );
			network.addLayer( layer );

			layer = new FullConnLayer( layer->getBaseOutDims(), target[ 0 ].size() );
			layer->setActFunc( ActFunc::softmax() );
			network.addLayer( layer );
		}

		gx_eval( "before train", network, input4eval, target4eval );

		network.print();

		bool ret = network.train( input, target, args );

		Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		//gx_eval( "after train", network, input4eval, target4eval );
	}

	//load model
	{
		Network network;

		Utils::load( path, &network );

		gx_eval( "load model", network, input4eval, target4eval );
	}
}

int main( const int argc, char * argv[] )
{
	CmdArgs_t defaultArgs = {
		.mThreadCount = 8,
		.mTrainingCount = 0,
		.mEvalCount = 0,
		.mEpochCount = 5,
		.mMiniBatchCount = 128,
		.mLearningRate = 2.0,
		.mLambda = 5.0,
		.mIsShuffle = true,
		.mIsDataAug = true,
	};

	CmdArgs_t args = defaultArgs;

	Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

