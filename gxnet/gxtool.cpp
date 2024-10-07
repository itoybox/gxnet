#include "network.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <set>
#include <float.h>

#include <unistd.h>
#include <syslog.h>
#include <getopt.h>

using namespace gxnet;

bool readImage( const char * path, DataVector * input )
{
	auto getNumberVector = []( std::string const & line, DataVector * data ) {
		const std::regex colon( "," );

		std::vector< std::string > srow{
			std::sregex_token_iterator( line.begin(), line.end(), colon, -1 ),
			std::sregex_token_iterator() };

		data->resize( srow.size(), 0 );
		for( size_t i = 0; i < srow.size(); i++ ) {
			( *data )[ i ] = std::stod( srow[ i ] ) / 255.0;
		}
	};

	std::ifstream fp( path );

	if( !fp ) {
		printf( "open %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	std::string line;

	if( ! std::getline( fp, line ) ) {
		printf( "getline %s fail, errno %d, %s\n", path, errno, strerror( errno ) );
		return false;
	}

	getNumberVector( '[' == line[ 0 ] ? line.c_str() + 1 : line, input );

	//printf( "%s read %s, size %zu\n", __func__, path, input->size() );

	return true;
}

int test( const char * model, const char * file )
{
	DataVector input, output;

	if( ! readImage( file, &input ) ) return -1;

	Network network;

	if( ! Utils::load( model, &network ) ) return -1;

	if( input.size() < network.getLayers()[ 0 ]->getBaseInSize() ) {
		DataVector newInput;
		Utils::expandMnistImage( input, &newInput );
		input = newInput;
	}

	bool ret = network.forward( input, &output );

	if( ! ret ) {
		printf( "forward fail\n" );
		return -1;
	}

	int result = Utils::max_index( std::begin( output ), std::end( output ) );

	printf( "%s    \t-> %d, nn.output %f\n", file, result, output[ result ] );

	return result;
}

int eval( const char * model, const char * images, const char * labels )
{
	DataMatrix input, target;

	if( ! Utils::loadMnistImages( 0, images, &input ) ) {
		printf( "read %s fail\n", images );
		return -1;
	}

	if( ! Utils::loadMnistLabels( 0, labels, &target ) ) {
		printf( "read %s fail\n", labels );
		return -1;
	}

	char result[ 1024 ] = { 0 };
	snprintf( result, sizeof( result ), "%s.result", images );

	FILE * fp = fopen( result, "w" );
	if( NULL == fp ) {
		printf( "open %s fail\n", result );
		return -1;
	}

	Network network;

	if( ! Utils::load( model, &network ) ) return -1;

	DataVector output;

	Dims inDims = { 28, 28 };

	for( size_t i = 0; i < input.size(); i++ ) {

		if( input[ i ].size() < network.getLayers()[ 0 ]->getBaseInSize() ) {
			DataVector newInput;
			Utils::expandMnistImage( input[ i ], &newInput );
			input[ i ] = newInput;
		}

		bool ret = network.forward( input[ i ], &output );

		if( ! ret ) {
			printf( "forward fail\n" );
			return -1;
		}

		int outputType = Utils::max_index( std::begin( output ), std::end( output ) );
		int targetType = Utils::max_index( std::begin( target[ i ] ), std::end( target[ i ] ) );

		fprintf( fp, "%d %d %.6f\n", targetType, outputType, output[ outputType ] );
	}

	printf( "save eval result in %s\n", result );

	fclose( fp );
	
	return 0;
}

void usage( const char * name )
{
	printf( "%s --model <model file> [ --file <mnist file> ] [ --images <idx3 ubyte> --labels <idx1 ubyte> ]\n", name );
}

int main( const int argc, char * argv[] )
{
	static struct option opts[] = {
		{ "model",   required_argument,  NULL, 1 },
		{ "file",  required_argument,  NULL, 2 },
		{ "images",  required_argument,  NULL, 3 },
		{ "labels",  required_argument,  NULL, 4 },
		{ 0, 0, 0, 0}
	};

	char * model = NULL, * file = NULL;
	char * images = NULL, * labels = NULL;

	int c = 0;
	while( ( c = getopt_long( argc, argv, "", opts, NULL ) ) != EOF ) {
		switch( c ) {
			case 1:
				model = optarg;
				break;
			case 2:
				file = optarg;
				break;
			case 3:
				images = optarg;
				break;
			case 4:
				labels = optarg;
				break;
			default:
				usage( argv[ 0 ] );
				break;
		}
	}

	if( ( NULL == model ) ||
		( ! ( ( NULL != file ) || ( NULL != images && NULL != labels ) ) )
	) {
		usage( argv[ 0 ] );
		return 0;
	}

	int ret = -1;

	if( NULL != file ) ret = test( model, file );

	if( NULL != images && NULL != labels ) ret = eval( model, images, labels );

	return ret;
}

