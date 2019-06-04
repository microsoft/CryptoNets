// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using HEWrapper;

namespace NeuralNetworks
{
    public class LLDuplicateLayer : BaseLayer
    {
        public ulong Count { get; set; }
        public override IMatrix Apply(IMatrix m)
        {
            IVector[] resArray = new IVector[m.ColumnCount];
            ParallelProcessInEnv( resArray.Length, (env, task, i) =>
            {
                resArray[i] = m.GetColumn(i).Duplicate(Count, env);
            });
            IMatrix res = Factory.GetMatrix(resArray, m.Format, CopyVectors:false);
            return res;
        }

        public override int OutputDimension()
        {
            int shift = 1;
            int inputDim = Source.OutputDimension();
            while (shift < inputDim) shift *= 2;
            return shift * (int)Count;
        }
    }
}
