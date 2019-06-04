// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using HEWrapper;
using System.Linq;

namespace NeuralNetworks
{
    public class LLPoolLayer : BaseLayer
    {
        ConvolutionEngine convolutionEngine = new ConvolutionEngine();
        double[] _Weights = null;
        public double[] Weights { get { return _Weights; } set { _Weights = value; layerPrepared = false; } }
        double[] _Bias = null;
        public double[] Bias { get { return _Bias; } set { _Bias = value; layerPrepared = false; } }
        public int[] InputShape { get { return convolutionEngine.InputShape; } set { convolutionEngine.InputShape = value; layerPrepared = false; } }
        public int[] KernelShape { get { return convolutionEngine.KernelShape; } set { convolutionEngine.KernelShape = value; layerPrepared = false; } }
        public double WeightsScale { get; set; } = 1.0;

        public override double GetOutputScale()
        {
            return ((Weights == null) ? Offsets.Length : WeightsScale) * Source.GetOutputScale();
        }
        public int[] Stride { get { return convolutionEngine.Stride; } set { convolutionEngine.Stride = value; layerPrepared = false; } }
        public bool[] Padding { get { return convolutionEngine.Padding; } set { convolutionEngine.Padding = value; layerPrepared = false; } }
        public int[] Upperpadding { get { return convolutionEngine.Upperpadding; } set { convolutionEngine.Upperpadding = value; layerPrepared = false; } }
        public int[] Lowerpadding { get { return convolutionEngine.Lowerpadding; } set { convolutionEngine.Lowerpadding = value; layerPrepared = false; } }
        public int[] MapCount { get { return convolutionEngine.MapCount; } set { convolutionEngine.MapCount = value; layerPrepared = false; } }
        private int kernelSize = -1; // the value -1 is used such that it will throw an exception if it was not computed

        IVector[] weightWindows = null;
        IVector[] biasVectors = null;
        int[][] Offsets { get { return convolutionEngine.Offsets; } }
        int[][] Corners { get { return convolutionEngine.Corners; } }

        public Vector<double> HotIndices { get; set; } = null;

        public override void Dispose()
        {
            if (weightWindows != null)
                foreach (var w in weightWindows)
                    if (w != null) w.Dispose();
            if (biasVectors != null)
                foreach (var b in biasVectors)
                    if (b != null) b.Dispose();
            weightWindows = null;
            biasVectors = null;


        }
        public override void Prepare()
        {
            if (!layerPrepared)
            {
                convolutionEngine.Prepare();
                kernelSize = KernelShape.Aggregate(1, (acc, val) => acc * val);
                if (Bias == null) kernelSize++;
                if (Weights == null) return;
                PrepareWeightsWindows();
                double BiasScale = GetOutputScale();
                int maps = (MapCount == null) ? 1 : MapCount.Aggregate(1, (acc, val) => acc * val);
                if (HotIndices == null)
                {
                    HotIndices = Vector<double>.Build.Dense(Corners.Length) + 1;
                }
                if (Bias != null)
                {
                    biasVectors = new IVector[maps];
                    ParallelProcessInEnv(maps, (env, taskIndex, mapIndex) =>
                    {
                        biasVectors[mapIndex] = Factory.GetPlainVector(HotIndices * Bias[mapIndex], EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
                    });
                }
                else
                {
                    biasVectors = new IVector[maps];
                    ParallelProcessInEnv(maps, (env, taskIndex, mapIndex) =>
                    {
                        biasVectors[mapIndex] = Factory.GetPlainVector(HotIndices * Weights[(mapIndex + 1) * kernelSize - 1], EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
                    });

                }

                layerPrepared = true;
            }
        }


        double ElementAt(double[] w, int[] Corner, int[] offset, int[] shape, int bias = 0)
        {
            var l = convolutionEngine.Location(Corner, offset, shape, bias);
            if (l < 0) return 0;
            return w[l];
        }




        void PrepareWeightsWindows()
        {
            int maps = (MapCount == null) ? 1 : MapCount.Aggregate(1, (acc, val) => acc * val);

            weightWindows = new IVector[maps];
            for (int m = 0; m < maps; m++)
            {
                var w = Offsets.Select(offset => ElementAt(Weights, null, offset, KernelShape, m * kernelSize));
                weightWindows[m] = Factory.GetPlainVector(Vector<double>.Build.DenseOfEnumerable(w), EVectorFormat.sparse, WeightsScale);
            }
        }

        public override IMatrix Apply(IMatrix m)
        {
            if (Weights == null) // pool without convolve
            {
                IVector vec = null;
                ProcessInEnv(env =>
               {


                   vec = Enumerable.Range(0, (int)m.ColumnCount).Select(i => m.GetColumn(i)).Aggregate((v1, v2) => v1.Add(v2, env));
                   vec.RegisterScale(vec.Scale * m.ColumnCount);
               });
                return Factory.GetMatrix(new IVector[] { vec }, EMatrixFormat.ColumnMajor, CopyVectors: false);
            }
            else
            {
                int maps = biasVectors.Length;
                IVector[] res = new IVector[maps];
                ParallelProcessInEnv(maps, (env, task, k) =>
               {
                   using (var mul = m.Mul(weightWindows[k], env))
                       res[k] = mul.Add(biasVectors[k], env);
               });
                return Factory.GetMatrix(res, EMatrixFormat.ColumnMajor, CopyVectors: false);
            }
        }


        public override int OutputDimension()
        {
            if (!layerPrepared) Prepare();
            int count = Corners.Length;
            if (Weights == null)
            {
                return count;
            }

            int maps = (MapCount == null) ? 1 : MapCount.Aggregate(1, (acc, val) => acc * val);
            return count * maps;

        }
    }
}
