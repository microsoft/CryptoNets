// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using HEWrapper;

namespace NeuralNetworks
{
    public class LLVectorizeLayer : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            var vec = ProcessInEnv( env =>
            {
                return m.ConvertToColumnVector(env);
            });
            return Factory.GetMatrix(new IVector[] { vec }, EMatrixFormat.ColumnMajor, CopyVectors: false);
        }

        public int OutputDim { get; set; } = -1;
        public override int OutputDimension()
        {
            return (OutputDim > 0) ? OutputDim : base.OutputDimension();
        }
    }
}
