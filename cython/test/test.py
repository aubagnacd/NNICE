# Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#-----------------------------------------------------------------------------
import numpy as np
from pyNNICE import pySnglInf

net = pySnglInf()
net.ImportNN(b"../../test/NNs/2x80_tanh/model.h5", b"../../test/NNs/2x80_tanh/architecture.json", b"", b"dense_layer_", b"", b"output_layer")

inputs = np.double([0.68134377, -1.18707218, 0.86916381, -1.22160812, 1.38525008, -1.28156564, -11.19841239, -10.24870305, -6.97707644])

res = net.run_ai(inputs)

print(res)