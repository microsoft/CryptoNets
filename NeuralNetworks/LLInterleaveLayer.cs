// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Linq;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLInterleaveLayer : BaseLayer
    {
		public int Shift { get; set; }
        public List<int> SelectedIndices { get; set; } = null;
        public int InputGrossDimension { get; set; } = -1;

        IVector mask;

        public override void Dispose()
        {
            if (mask != null) mask.Dispose();
            mask = null;
        }
        public override void Prepare()
        {
            ProcessInEnv(env =>
            {
                if (InputGrossDimension < 0) InputGrossDimension = SelectedIndices.Max() + 1;
                var maskVector = Vector<Double>.Build.DenseOfIndexed(InputGrossDimension,
                    SelectedIndices.Select(i => new Tuple<int, double>(i, 1.0)));
                mask = Factory.GetPlainVector(maskVector, EVectorFormat.dense, 1);

            });
        }
        public override IMatrix Apply(IMatrix m)
        {
            IVector[] clean = new IVector[m.ColumnCount];
            IVector interleaved = null;
            ParallelProcessInEnv(clean.Length, (env, task, i) =>
                clean[i] = m.GetColumn(i).PointwiseMultiply(mask, env));
            using (var cleanMat = Factory.GetMatrix(clean, EMatrixFormat.ColumnMajor, CopyVectors: false))
            {
                ProcessInEnv(env =>
                {
                    interleaved = cleanMat.Interleave(Shift, env);
                });
            }
            return Factory.GetMatrix(new IVector[] { interleaved }, EMatrixFormat.ColumnMajor, CopyVectors: false);
        }

        public override int OutputDimension()
        {
            return InputGrossDimension;
        }
    }
}
