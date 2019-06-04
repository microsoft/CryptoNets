// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using HEWrapper;

namespace NeuralNetworks
{
    public class SquareActivation : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            return(ProcessInEnv( env => m.ElementWiseMultiply(m, env)));
        }

        public override double GetOutputScale()
        {
            var s = Source.GetOutputScale();
            return s * s;
        }
    }
}
