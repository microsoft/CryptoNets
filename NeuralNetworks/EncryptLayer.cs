// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class EncryptLayer : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            IMatrix res = null;
            var mr = m as RawMatrix;
            ProcessInEnv( env => res = Factory.GetEncryptedMatrix((Matrix<Double>)mr.Data, EMatrixFormat.ColumnMajor, 1));
            res.RegisterScale(m.Scale);
            return res;
        }

    }
}
