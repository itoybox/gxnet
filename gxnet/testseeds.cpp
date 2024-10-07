#include "utils.h"
#include "network.h"
#include "activation.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <set>
#include <limits.h>
#include <float.h>

#include <unistd.h>
#include <stdio.h>

using namespace gxnet;

/*
* Load comma separated values from file and normalize the values
*/
bool loadData( const char * filename, DataMatrix * data, std::set< int > * labels )
{
	const std::regex comma(",");

	std::ifstream fp( filename );

	if( !fp ) return false;

	std::string line;

	while( fp && std::getline( fp, line ) ) {
		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), comma, -1 ),
			std::sregex_token_iterator() };

		data->emplace_back( DataVector( srow.size() ) );

		std::transform( srow.begin(), srow.end(), std::begin( data->back() ),
				[](std::string const& val) {return std::stof(val); } );

		labels->insert( std::stoi( srow.back() ) );
	}

	DataVector min, max;
	min.resize( data[ 0 ].size(), FLT_MAX );
	max.resize( data[ 0 ].size(), 1.0 * INT_MIN );

	// normalize data
	{
		for( auto & item : * data ) {
			for( size_t i = 0; i < item.size(); i++ ) {
				if( item[ i ] > max[ i ] ) max[ i ] = item[ i ];
				if( item[ i ] < min[ i ] ) min[ i ] = item[ i ];
			}
		}

		for( auto & item : * data ) {
			// skip last element, it's the label
			for( size_t i = 0; i < item.size() - 1; i++ ) {
				item[ i ] = ( item[ i ] - min[ i ] ) / ( max[ i ] - min[ i ] );
			}
		}
	}

	return true;
}

void check( const char * tag, Network & network, DataMatrix & input, DataMatrix & target )
{
	if( gx_is_inner_debug ) network.print();

	int correct = 0;

	for( size_t i = 0; i < input.size(); i++ ) {

		DataVector output;

		bool ret = network.forward( input[ i ], &output );

		int outputType = Utils::max_index( std::begin( output ), std::end( output ) );
		int targetType = Utils::max_index( std::begin( target[ i ] ), std::end( target[ i ] ) );

		if( gx_is_inner_debug ) printf( "forward %d, index %zu, %d %d\n", ret, i, outputType, targetType );

		if( outputType == targetType ) correct++;

		for( size_t j = 0; gx_is_inner_debug && j < output.size(); j++ ) {
			printf( "\t%zu %.8f %.8f\n", j, output[ j ], target[ i ][ j ] );
		}
	}

	printf( "Network %s, %d/%ld = %.2f\n", tag, correct, input.size(), ((float)correct) / input.size() );
}

void splitData( const CmdArgs_t & args, const DataMatrix & data, const std::set< int > & labels,
		DataMatrix * input, DataMatrix * target,
		DataMatrix * input4eval, DataMatrix * target4eval )
{
	std::vector< int > idxOfData( data.size() );
	std::iota( idxOfData.begin(), idxOfData.end(), 0 );

	std::map< int, int > mapOflabels;
	for( auto & item : labels ) mapOflabels[ item ] = mapOflabels.size();

	for( int i = args.mEvalCount; i > 0; i-- ) {
		int n = std::rand() % idxOfData.size();

		if( gx_is_inner_debug ) n = i - 1;

		const DataVector & item = data[ idxOfData[ n ] ];

		// remove last element, it's the label
		input4eval->emplace_back( DataVector( item.size() - 1 ) );
		std::copy( std::begin( item ), std::end( item ) - 1, std::begin( input4eval->back() ) );

		target4eval->emplace_back( DataVector() );
		target4eval->back().resize( mapOflabels.size(), 0 );
		target4eval->back()[ mapOflabels[ item[ item.size() - 1 ] ] ] = 1;

		idxOfData.erase( idxOfData.begin() + n );
	}

	for( size_t i = 0; i < idxOfData.size(); i++ ) {

		const DataVector & item = data[ idxOfData[ i ] ];

		// remove last element, it's the label
		input->emplace_back( DataVector( item.size() - 1 ) );
		std::copy( std::begin( item ), std::end( item ) - 1, std::begin( input->back() ) );

		target->emplace_back( DataVector() );
		target->back().resize( mapOflabels.size(), 0 );
		target->back()[ mapOflabels[ item[ item.size() - 1 ] ] ] = 1;

		if( args.mTrainingCount > 0 && (int)input->size() >= args.mTrainingCount ) break;
	}
}

void test( const CmdArgs_t & args )
{
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	DataMatrix data;
	std::set< int > labels;

	loadData( "seeds_dataset.csv", &data, &labels );

	DataMatrix input, target, input4eval, target4eval;

	splitData( args, data, labels, &input, &target, &input4eval, &target4eval );

	Dims baseInDims = { input[ 0 ].size() };

	const char * path = "./seeds.model";

	// train & check & save
	{
		Network network;

		network.setLossFuncType( Network::eCrossEntropy );

		BaseLayer * layer = NULL;

		layer = new FullConnLayer( baseInDims, input[ 0 ].size() );
		layer->setActFunc( ActFunc::sigmoid() );
		network.addLayer( layer );

		layer = new FullConnLayer( layer->getBaseOutDims(), target[ 0 ].size() );
		layer->setActFunc( ActFunc::softmax() );
		network.addLayer( layer );

		check( "before train", network, input4eval, target4eval );

		network.print( true );

		bool ret = network.train( input, target, args );

		Utils::save( path, network );

		printf( "train %s\n", ret ? "succ" : "fail" );

		check( "after train", network, input4eval, target4eval );
	}

	{
		Network network;

		Utils::load( path, &network );

		network.print();

		check( "load model", network, input4eval, target4eval );
	}
}

int main( const int argc, char * argv[] )
{
	CmdArgs_t defaultArgs = {
		.mThreadCount = 2,
		.mEvalCount = 42,
		.mEpochCount = 10,
		.mMiniBatchCount = 2,
		.mLearningRate = 0.1,
		.mIsShuffle = true,
	};

	CmdArgs_t args = defaultArgs;

	Utils::getCmdArgs( argc, argv, defaultArgs, &args );

	test( args );

	return 0;
}

