
#include "layer.h"
#include "context.h"
#include "optim.h"

#include "utils.h"

#include <cstdio>
#include <typeinfo>
#include <memory>

using namespace gxnet;

template< typename TConvLayer >
void testConvLayer()
{
	printf( "========== test typename %s ==========\n", typeid( TConvLayer ).name() );

	DataVector biases = { 1, 1 };

	MDVector filters;
	filters.second = { 2, 2, 3, 3 };
	filters.first.resize( gx_dims_flatten_size( filters.second ) );

	for( size_t i = 0; i < filters.first.size(); i++ ) filters.first[ i ] = i * 0.1;

	MDVector inMD;
	inMD.second = { 2, 2, 4, 4, };
	inMD.first.resize( gx_dims_flatten_size( inMD.second ) );
	std::iota( std::begin( inMD.first ), std::end( inMD.first ), 0 );

	Utils::printMDVector( "input", inMD );

	Utils::printMDVector( "filters", filters );

	TConvLayer conv( { 2, 4, 4, }, filters, biases );

	conv.print( true );

	std::unique_ptr< BaseLayerContext > ctx( conv.createCtx() );

	ctx->setInMD( &inMD );

	conv.forward( ctx.get() );

	Utils::printMDVector( "conv.output", ctx->getOutMD() );
	Utils::printVector( "conv.output", ctx->getOutMD().first );

	DataVector inDelta( inMD.first.size() );

	MDVector & deltaData = ctx->getDeltaMD();

	for( size_t i = 0; i < deltaData.first.size(); i++ ) deltaData.first[ i ] = i * 0.1;

	conv.backward( ctx.get(), &inDelta );

	Utils::printMDVector( "conv.outDelta", ctx->getDeltaMD() );

	Utils::printVector( "conv.inDelta", inDelta, inMD.second );
	Utils::printVector( "conv.inDelta", inDelta );

	conv.collectGradients( ctx.get() );

	Utils::printVector( "filters.grad", ctx->getGradients()[ 0 ], filters.second );

	std::unique_ptr< Optim > optim( Optim::SGD( 0.1, 1 ) );

	conv.applyGradients( *( ctx.get() ), optim.get(), 1, 1 );

	conv.print( true );
}

void testMaxPoolLayer()
{
	Dims inDims = { 2, 1, 4, 4 };

	DataVector input( gx_dims_flatten_size( inDims ) );
	std::iota( std::begin( input ), std::end( input ), 0 );

	MDVector inMD( input, inDims );

	Utils::printMDVector( "input", inMD );

	MaxPoolLayer maxpool( { 1, 4, 4 }, 2 );

	std::unique_ptr< BaseLayerContext > ctx( maxpool.createCtx() );
	ctx->setInMD( &inMD );

	maxpool.forward( ctx.get() );

	Utils::printMDVector( "maxpool.output", ctx->getOutMD() );

	DataVector inDelta( input.size() );

	ctx->getDeltaMD().first = 0.5;

	maxpool.backward( ctx.get(), &inDelta );

	Utils::printVector( "maxpool.inDelta", inDelta, ctx->getInMD().second );
}

int main( int argc, const char * argv[] )
{
	gx_is_inner_debug = true;

	//testConvLayer<ConvLayer>();

	testConvLayer<ConvExLayer>();

	//testMaxPoolLayer();

	return 0;
}

