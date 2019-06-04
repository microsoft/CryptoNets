// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLPackedDenseLayer : BaseLayer
    {
        double[] _Weights = null;
        public double[] Weights { get { return _Weights; } set { _Weights = value; layerPrepared = false; } }
        double[] _Bias = null;
        public double[] Bias { get { return _Bias; } set { _Bias = value; layerPrepared = false; } }
        public double WeightsScale { get; set; } = 1.0;

        public ulong PackingCount { get; set; } = 1;
        public int PackingShift { get; set; } = 0;

        public override double GetOutputScale()
        {
            return WeightsScale * Source.GetOutputScale();
        }

        IMatrix WeightsMatrix = null;
        IMatrix BiasMatrix = null;

        public override void Dispose()
        {
            if (WeightsMatrix != null) WeightsMatrix.Dispose();
            if (BiasMatrix != null) BiasMatrix.Dispose();
            WeightsMatrix = null;
            BiasMatrix = null;
        }

        public override void Prepare()
        {
            if (layerPrepared) return;
            int maps = Bias.Length;
            int mapLength = Weights.Length / Bias.Length;
            ////////////////////////////
            var ColumnWeightMatrix = Matrix<Double>.Build.DenseOfRowMajor(Bias.Length, Weights.Length / Bias.Length, Weights);
            var NewRowsCount = (int)((maps + (int)PackingCount - 1) / (int)PackingCount);
            var StackedMatrix = Matrix<double>.Build.Dense(NewRowsCount, (int)PackingCount * PackingShift);
            var PaddedBias = Matrix<double>.Build.Dense(NewRowsCount, (int)PackingCount * PackingShift);
            for (int i = 0; i < maps; i++)
            {
                int col = i % (int)PackingCount;
                int row = i / (int)PackingCount;
                var mat = Matrix<double>.Build.DenseOfRowVectors(new Vector<double>[] { ColumnWeightMatrix.Row(i) });
                StackedMatrix.SetSubMatrix(row, col * PackingShift, mat);
                PaddedBias[row, (col + 1) * PackingShift - 1] = Bias[i];
            }

            BiasMatrix = Factory.GetPlainMatrix(PaddedBias, EMatrixFormat.RowMajor, Source.GetOutputScale() * WeightsScale);
            WeightsMatrix = Factory.GetPlainMatrix(StackedMatrix, EMatrixFormat.RowMajor, WeightsScale);
        }
        public override int OutputDimension()
        {
            return Bias.Length;
        }

        public override IMatrix Apply(IMatrix m)
        {
            if (m.ColumnCount > 1) throw new Exception("Expecting only one column");
            IVector[] res = new IVector[WeightsMatrix.RowCount];
            var vector = m.GetColumn(0);
            ParallelProcessInEnv(res.Length, (env, task, k) =>
            {
                using (var mul = WeightsMatrix.GetRow(k).DotProduct(vector, (ulong)PackingShift, env))
                    res[k] = mul.Add(BiasMatrix.GetRow(k), env);
            });
            return Factory.GetMatrix(res, EMatrixFormat.ColumnMajor, CopyVectors: false);
        }
    }
}
