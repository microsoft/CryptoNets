// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using HEWrapper;
using System;
using System.Linq;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class PoolLayer : BaseLayer
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
        private ConcurrentBag<IVector> TempVectors = new ConcurrentBag<IVector>();
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
            if (layerPrepared) return;
            convolutionEngine.Prepare();

            kernelSize = KernelShape.Aggregate(1, (acc, val) => acc * val);
            if (Bias == null) kernelSize++;
            if (Weights == null) return;
            PrepareWeightsWindows();
            biasVectors = null;
            layerPrepared = true;
            

        }


        IVector ElementAt(IMatrix m, int[] Corner, int[] offset, int[] shape, int bias = 0)
        {
            var l = convolutionEngine.Location(Corner, offset, shape, bias);
            if (l < 0)
            {
                var t = Vector<double>.Build.DenseOfArray(new double[m.RowCount]);
                var z = (m.IsEncrypted) ? Factory.GetEncryptedVector(t, EVectorFormat.dense, m.Scale) :
                    Factory.GetPlainVector(t, EVectorFormat.dense, m.Scale);
                TempVectors.Add(z);
                return z;
            }
            return m.GetColumn(l);
        }

        void ReleaseTemp()
        {
            Parallel.ForEach(TempVectors, v => v.Dispose());
            lock(this) // clear the list of temp-vectors
            {
                TempVectors = new ConcurrentBag<IVector>(); 
            }
        }

        double ElementAt(double[] w, int[] corner, int[] offset, int[] shape, int bias = 0)
        {
            var l = convolutionEngine.Location(corner, offset, shape, bias);
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

        IVector ConvolveOnce(IMatrix m, int cornerIndex, int mapIndex, IComputationEnvironment env)
        {
            var corner = Corners[cornerIndex];

            // patch matrix should not be disposed since this will result in deleting the inputs which are copied only by reference
            var patch = Factory.GetMatrix(Offsets.Select(offset => ElementAt(m, corner, offset, InputShape)).ToArray(), EMatrixFormat.ColumnMajor, CopyVectors: false);
            patch.DataDisposedExternaly = true;
            return patch.Mul(weightWindows[mapIndex], env);
        }

        /// this is for a pool layer which is a convolution but with all the weights equal 1
        /// the scale of the result is normalized such that it will act as if an average was performed (as opposed to sum)
        IVector SumOnce(IMatrix m, int[] corner) 
        {
            IVector agg = null;
            var res = ProcessInEnv( env =>
            {
                foreach (var offset in Offsets)
                {
                    var element = ElementAt(m, corner, offset, InputShape);
                    if (element != null)
                    {
                        var t = agg;
                        agg = (agg == null) ? element : agg.Add(element, env);
                        if (t != null) t.Dispose();
                    }
                }
                agg.RegisterScale(agg.Scale * Offsets.Length);
                return agg;
            });
            return res;

        }



        public override IMatrix Apply(IMatrix m)
        {
            if (Weights == null) // pool without convolve
            {
                var res = Factory.GetMatrix(Corners.Select(Corner => SumOnce(m, Corner)).ToArray(), EMatrixFormat.ColumnMajor, CopyVectors: false);
                ReleaseTemp();
                return res;
            }
            else
            {
                int maps = (MapCount == null) ? 1 : MapCount.Aggregate(1, (acc, val) => acc * val);
                if (Bias != null)
                {
                    if (biasVectors == null || biasVectors[0].Dim != m.RowCount)
                    {
                        if (biasVectors != null)
                            foreach (var b in biasVectors)
                                if (b != null) b.Dispose();

                        biasVectors = new IVector[maps];
                        ParallelProcessInEnv(maps, (env, taskIndex, mapIndex) =>
                        {
                            var t = Enumerable.Range(0, (int)m.RowCount).Select(i => Bias[mapIndex]);
                            biasVectors[mapIndex] = Factory.GetPlainVector(Vector<double>.Build.DenseOfEnumerable(t), EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
                        });
                    }


                    var res = new IVector[maps * Corners.Length];
                    Console.WriteLine("Pool (bias) layer with {0} maps and {1} locations (total size {2}) kernel size {3}",
                        maps, Corners.Length, res.Length, kernelSize);


                    ParallelProcessInEnv(maps * Corners.Length, (env, currentTask, k) =>
                   {

                       int mapIndex = k / Corners.Length;
                       int CornerIndex = k - (mapIndex * Corners.Length);
                       using (var conv = ConvolveOnce(m, CornerIndex, mapIndex, env))
                           res[k] = conv.Add(biasVectors[mapIndex], env);
                       if (k % 17 == 0) Console.Write("Done (bias) {0}\r", k);

                   });
                    var mat = Factory.GetMatrix(res, EMatrixFormat.ColumnMajor, CopyVectors: false);
                    ReleaseTemp();
                    return mat;
                }
                else
                {
                    if (biasVectors == null || biasVectors[0].Dim != m.RowCount)
                    {
                        if (biasVectors != null)
                            foreach (var b in biasVectors)
                                if (b != null) b.Dispose();

                        biasVectors = new IVector[maps];
                        ParallelProcessInEnv(maps, (env, taskIndex, mapIndex) =>
                       {
                           var t = Enumerable.Range(0, (int)m.RowCount).Select(i => Weights[(mapIndex + 1) * kernelSize - 1]);
                           biasVectors[mapIndex] = Factory.GetPlainVector(Vector<double>.Build.DenseOfEnumerable(t), EVectorFormat.dense, Source.GetOutputScale() * WeightsScale);
                       });
                    }
                    var res = new IVector[maps * Corners.Length];
                    Console.WriteLine("Pool (no-bias) layer with {0} maps and {1} locations (total size {2}) kernel size {3}",
                        maps, Corners.Length, res.Length, kernelSize);
                    ParallelProcessInEnv(res.Length, (env, currentTask, k) =>
                   {
                       int mapIndex = k / Corners.Length;
                       int CornerIndex = k - (mapIndex * Corners.Length);
                       using (var conv = ConvolveOnce(m, CornerIndex, mapIndex, env))
                           res[k] = conv.Add(biasVectors[mapIndex], env);
                       if (k % 17 == 0) Console.Write("Done (bias) {0}\r", k);


                   });
                    IMatrix mat = Factory.GetMatrix(res, EMatrixFormat.ColumnMajor, CopyVectors: false);
                    ReleaseTemp();
                    return mat;
                }
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
