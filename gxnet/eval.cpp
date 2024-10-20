
#include "eval.h"
#include "utils.h"

namespace gxnet {

void gx_eval( const char * tag, Network & network, DataMatrix & input, DataMatrix & target )
{
	printf( "%s( %s, ..., input { %ld }, target { %ld } )\n", __func__, tag, input.size(), target.size() );

	if( gx_is_inner_debug ) network.print();
	network.setTraining( false );

	DataMatrix confusionMatrix;
	DataVector targetTotal;

	size_t maxClasses = target[ 0 ].size();
	confusionMatrix.resize( maxClasses );
	targetTotal.resize( maxClasses );
	for( size_t i = 0; i < maxClasses; i++ ) confusionMatrix[ i ].resize( maxClasses, 0.0 );

	int correct = 0;

	DataMatrix output;

	bool ret = network.forward( input, &output );

	if( ! ret ) {
		printf( "forward fail\n" );
		return;
	}

	for( size_t i = 0; i < output.size(); i++ ) {

		int outputType = Utils::max_index( std::begin( output[ i ] ), std::end( output[ i ] ) );
		int targetType = Utils::max_index( std::begin( target[ i ] ), std::end( target[ i ] ) );

		if( gx_is_inner_debug ) printf( "forward %d, index %zu, %d %d\n", ret, i, outputType, targetType );

		if( outputType == targetType ) correct++;

		confusionMatrix[ targetType ][ outputType ] += 1;
		targetTotal[ targetType ] += 1;

		for( size_t j = 0; gx_is_inner_debug && j < output[ i ].size() && j < 10; j++ ) {
			printf( "\t%zu %.8f %.8f\n", j, output[ i ][ j ], target[ i ][ j ] );
		}
	}

	printf( "check %s, %d/%ld = %.2f\n", tag, correct, input.size(), ((float)correct) / input.size() );

	for( size_t i = 0; i < confusionMatrix.size(); i++ ) {
		for( auto & item : confusionMatrix[ i ] ) item = item / ( targetTotal[ i ] ? targetTotal[ i ] : 1 );
	}

	Utils::printMatrix( "confusion matrix", confusionMatrix, false, true );
}


}; // namespace gxnet;

