// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Linq;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLInterleavedDenseLayer : BaseLayer
    {
        public double[] Weights { get; set; }
        public double[] Bias { get; set; }
        public int WeightsScale { get; set; }
        public int Shift { get; set; }
        public List<int> SelectedIndices { get; set; }

        IMatrix WeightsMatrix;
        IVector BiasVector;

        public override void Dispose()
        {
            if (WeightsMatrix != null) WeightsMatrix.Dispose();
            if (BiasVector != null) BiasVector.Dispose();
            WeightsMatrix = null;
            BiasVector = null;
        }
        public override IMatrix Apply(IMatrix m)
        {
            IVector v = null;
            ProcessInEnv(env =>
            {
                using (var mul = WeightsMatrix.Mul(m.GetColumn(0), env))
                {
                    v = mul.Add(BiasVector, env);
                }
            });
            return Factory.GetMatrix(new IVector[] { v }, EMatrixFormat.ColumnMajor, CopyVectors: false);
        }

        public override int OutputDimension()
        {
            return Bias.Length;
        }

        public override double GetOutputScale()
        {
            return Source.GetOutputScale() * WeightsScale;
        }

        IEnumerable<int> TargetIndices(int count)
        {
            int offset = 0;
            while(count>0)
            {
                for (int i = 0; (i < SelectedIndices.Count) && (count > 0); i++, count--)
                    yield return SelectedIndices[i] + offset;
                offset += Shift;
            }
        }

        public override void Prepare()
        {
            int columns = Weights.Length / Bias.Length;
            var smallWeightsMatrix = Matrix<Double>.Build.DenseOfRowMajor(Bias.Length, columns, Weights);
            var bigWeightsMatrix = Matrix<Double>.Build.Sparse(Bias.Length, Source.OutputDimension());
            var targetIndices = TargetIndices(columns).ToArray();
            for (int i = 0; i < columns; i++)
            {
                bigWeightsMatrix.SetColumn(targetIndices[i], smallWeightsMatrix.Column(i));
            }
            BiasVector = Factory.GetPlainVector(Vector<double>.Build.DenseOfArray(Bias), EVectorFormat.sparse, GetOutputScale());
            WeightsMatrix = Factory.GetPlainMatrix(bigWeightsMatrix, EMatrixFormat.RowMajor, WeightsScale);
        }
    }
}
