// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLDenseLayer : BaseLayer
    {
        double[] _Weights = null;
        public double[] Weights { get { return _Weights; } set { _Weights = value; layerPrepared = false; } }
        double[] _Bias = null;
        public double[] Bias { get { return _Bias; } set { _Bias = value; layerPrepared = false; } }
        public double WeightsScale { get; set; } = 1.0;
        public EVectorFormat InputFormat { get; set; } = EVectorFormat.dense;

        public bool ForceDenseFormat { get; set; } = false;


        public override double GetOutputScale()
        {
            return WeightsScale * Source.GetOutputScale();
        }

        IMatrix WeightsMatrix = null;
        IVector BiasVector = null;

        public override void Dispose()
        {
            if (WeightsMatrix != null) WeightsMatrix.Dispose();
            if (BiasVector != null) BiasVector.Dispose();
            WeightsMatrix = null;
            BiasVector = null;
        }
        public override void Prepare()
        {
            if (layerPrepared) return;
            if (ForceDenseFormat && InputFormat == EVectorFormat.sparse) throw new Exception("forcing dense format is only available when the input is dense");
            int maps = Bias.Length;

            if (InputFormat == EVectorFormat.dense)
            {
                BiasVector = Factory.GetPlainVector(Vector<Double>.Build.DenseOfArray(Bias), (ForceDenseFormat) ? EVectorFormat.dense : EVectorFormat.sparse, Source.GetOutputScale() * WeightsScale);
                WeightsMatrix = Factory.GetPlainMatrix(Matrix<Double>.Build.DenseOfRowMajor(Bias.Length, Weights.Length / Bias.Length, Weights), EMatrixFormat.RowMajor, WeightsScale);
            }
            else
            {
                BiasVector = Factory.GetPlainVector(Vector<Double>.Build.DenseOfArray(Bias), EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
                WeightsMatrix = Factory.GetPlainMatrix(Matrix<Double>.Build.DenseOfRowMajor(Bias.Length, Weights.Length / Bias.Length, Weights), EMatrixFormat.ColumnMajor, WeightsScale);
            }

            base.Prepare();

        }

        public override int OutputDimension()
        {
            return Bias.Length;
        }

        public override IMatrix Apply(IMatrix m)
        {
            if (m.ColumnCount > 1) throw new Exception("Expecting only one column");
            return ProcessInEnv( env =>
            {
                using (var mul = WeightsMatrix.Mul(m.GetColumn(0), env, ForceDenseFormat))
                {
                    var res = mul.Add(BiasVector, env);
                    var mat = Factory.GetMatrix(new IVector[] { res }, EMatrixFormat.ColumnMajor, CopyVectors: false);
                    return mat;
                }
            });
        }
    }
}
