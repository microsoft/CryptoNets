// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using NeuralNetworks;
using HEWrapper;

namespace NeuralNetworksTest
{
    public class FakeLayer : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            throw new NotImplementedException();
        }

        public override double GetOutputScale()
        {
            return 1.0;
        }
    }
}
