#pragma once

#include "common.h"
#include "network.h"

namespace gxnet {

class Network;

void gx_eval( const char * tag, Network & network, DataMatrix & input, DataMatrix & target, bool isDebug );


}; // namespace gxnet;

