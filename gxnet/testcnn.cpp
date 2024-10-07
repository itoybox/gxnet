
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

	Dims filterDims = { 2, 2, 3, 3 };
	DataVector filters( gx_dims_flatten_size( filterDims ) );
	for( size_t i = 0; i < filters.size(); i++ ) filters[ i ] = i * 0.1;

	Dims inDims = { 2, 2, 4, 4 };
	DataVector input( gx_dims_flatten_size( inDims ) );
	std::iota( std::begin( input ), std::end( input ), 0 );

	MDSpanRO inMS( input, inDims );

	Utils::printMDSpan( "input", inMS );

	Utils::printVector( "filters", filters );

	TConvLayer conv( { 2, 4, 4, }, filters, filterDims, biases );

	conv.print( true );

	std::unique_ptr< BaseLayerContext > ctx( conv.createCtx() );

	ctx->setInMS( &inMS );

	conv.forward( ctx.get() );

	Utils::printMDSpan( "conv.output", ctx->getOutRO() );
	Utils::printVector( "conv.output", ctx->getOutRO().data() );

	DataVector inDelta( input.size() );

	MDSpanRW & deltaMS = ctx->getDeltaMS();

	for( size_t i = 0; i < deltaMS.data().size(); i++ ) deltaMS.data()[ i ] = i * 0.1;

	conv.backward( ctx.get(), &inDelta );

	Utils::printMDSpan( "conv.outDelta", ctx->getDeltaRO() );

	Utils::printVector( "conv.inDelta", inDelta, inDims );
	Utils::printVector( "conv.inDelta", inDelta );

	conv.collectGradients( ctx.get() );

	Utils::printVector( "filters.grad", ctx->getGradients()[ 0 ], filterDims );

	std::unique_ptr< Optim > optim( Optim::SGD( 0.1, 1 ) );

	conv.applyGradients( *( ctx.get() ), optim.get(), 1, 1 );

	conv.print( true );
}

void testMaxPoolLayer()
{
	Dims inDims = { 2, 1, 4, 4 };

	DataVector input( gx_dims_flatten_size( inDims ) );
	std::iota( std::begin( input ), std::end( input ), 0 );

	MDSpanRO inMS( input, inDims );

	Utils::printMDSpan( "input", inMS );

	MaxPoolLayer maxpool( { 1, 4, 4 }, 2 );

	std::unique_ptr< BaseLayerContext > ctx( maxpool.createCtx() );
	ctx->setInMS( &inMS );

	maxpool.forward( ctx.get() );

	Utils::printMDSpan( "maxpool.output", ctx->getOutRO() );

	DataVector inDelta( input.size() );

	ctx->getDeltaMS().data() = 0.5;

	maxpool.backward( ctx.get(), &inDelta );

	Utils::printVector( "maxpool.inDelta", inDelta, ctx->getInMS().dims() );
}

int main( int argc, const char * argv[] )
{
	gx_is_inner_debug = true;

	//testConvLayer<ConvLayer>();

	testConvLayer<ConvExLayer>();

	//testMaxPoolLayer();

	return 0;
}

