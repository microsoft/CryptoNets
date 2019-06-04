// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLPreConvLayer : BaseLayer
    {
        ConvolutionEngine convolutionEngine = new ConvolutionEngine();
        public int[] InputShape { get { return convolutionEngine.InputShape; } set { convolutionEngine.InputShape = value; layerPrepared = false; } }
        public int[] KernelShape { get { return convolutionEngine.KernelShape; } set { convolutionEngine.KernelShape = value; layerPrepared = false; } }
        public int[] Stride { get { return convolutionEngine.Stride; } set { convolutionEngine.Stride = value; layerPrepared = false; } }
        public bool[] Padding { get { return convolutionEngine.Padding; } set { convolutionEngine.Padding = value; layerPrepared = false; } }
        public int[] Upperpadding { get { return convolutionEngine.Upperpadding; } set { convolutionEngine.Upperpadding = value; layerPrepared = false; } }
        public int[] Lowerpadding { get { return convolutionEngine.Lowerpadding; } set { convolutionEngine.Lowerpadding = value; layerPrepared = false; } }

        public bool[] UseAxisForBlocks { get; set; } = null;

        int outputDim = -1;
        int[][] shifts;
        IVector[][] masks;
        Vector<double> _HotIndices;
        public Vector<double> HotIndices { get { if (!layerPrepared) Prepare(); return _HotIndices; } private set { _HotIndices = value; } }
        
        IEnumerable<int> BlockOffset()
        {
            int[] block = new int[Stride.Length];
            int[] shifts = new int[Stride.Length];
            shifts[0] = 1;
            for (int i = 1; i < shifts.Length; i++) shifts[i] = shifts[i - 1] * InputShape[i - 1];
            int offset = 0;
            bool goodToGo = false;
            do
            {
                yield return offset;
                goodToGo = false;
                for (int i = 0; i < block.Length; i++)
                {
                    if (!UseAxisForBlocks[i]) continue;
                    block[i]++;
                    offset += shifts[i];
                    if (block[i] < Stride[i])
                    {
                        goodToGo = true;
                        break;
                    } else
                    {
                        offset -= block[i] * shifts[i];
                        block[i] = 0;
                    }
                }
            } while (goodToGo);

        }

        int[] CornersMap;

        public override void Dispose()
        {
            if (masks != null)
                foreach (var w in masks)
                    if (w != null)
                        foreach (var v in w)
                            if (v != null) v.Dispose();
            masks = null;
        }

        public override void Prepare()
        {
            if (!layerPrepared)
            {
                convolutionEngine.Prepare();
                if (UseAxisForBlocks == null) UseAxisForBlocks = InputShape.Select(x => true).ToArray();
                var len = convolutionEngine.Offsets.Length;
                int dim = convolutionEngine.InputShape.Aggregate((a, b) => a * b);
                var blockOffsets = BlockOffset().ToArray();
                var CornersProjections = convolutionEngine.Corners.Select(x => x[0]).Distinct().ToArray();
                var expectedBlockSize = CornersProjections.Length / (double)blockOffsets.Length;
                var smallBlockSize = (int)Math.Floor(expectedBlockSize);
                var largeBlockSize = (int)Math.Ceiling(expectedBlockSize);
                var numberOfLargeBlocks = CornersProjections.Length - blockOffsets.Length * smallBlockSize;
                CornersMap = Enumerable.Range(0, convolutionEngine.Corners.Length).Select(x => -1).ToArray();
                masks = new IVector[len][];
                shifts = new int[len][];
                for (int i = 0; i < len; i++)
                {
                    List<int>[] selections = Enumerable.Range(0, blockOffsets.Length).Select(t => new List<int>()).ToArray();
                    masks[i] = new IVector[blockOffsets.Length];
                    shifts[i] = new int[blockOffsets.Length];
                    for (int j = 0; j < shifts[i].Length; j++)
                    {
                        var thisBlockSize = (j > numberOfLargeBlocks) ? smallBlockSize : largeBlockSize;
                        shifts[i][j] = (j == 0) ? convolutionEngine.Location(null, convolutionEngine.Offsets[i], convolutionEngine.InputShape) : shifts[i][j - 1] + blockOffsets[j - 1] - blockOffsets[j] + thisBlockSize * Stride[0] * dim / InputShape[0];
                    }
                    for (int j = 0; j < convolutionEngine.Corners.Length; j++)
                    {
                        var location = convolutionEngine.Location(convolutionEngine.Corners[j], convolutionEngine.Offsets[i], convolutionEngine.InputShape);
                        var CornerID = (convolutionEngine.Corners[j][0] - convolutionEngine.Corners[0][0]) / Stride[0];
                        var block = (CornerID < largeBlockSize * numberOfLargeBlocks) ? CornerID / largeBlockSize : numberOfLargeBlocks + ((CornerID - largeBlockSize * numberOfLargeBlocks) / smallBlockSize);
                        if (location >= 0)
                        {
                            selections[block].Add(location);
                            var map = location - shifts[i][block];
                            if (CornersMap[j] >= 0 && CornersMap[j] != map)
                                throw new Exception("Internal Error");
                            CornersMap[j] = map;
                        }
                    }
                    ParallelProcessInEnv(masks[i].Length, (env, id, j) =>
                    {
                        if (selections[j].Any())
                        {
                            var v = Vector<double>.Build.SparseOfIndexed(dim, selections[j].Select(l => new Tuple<int, double>(l, 1.0)));
                            masks[i][j] = Factory.GetPlainVector(v, EVectorFormat.dense, 1);
                        }
                        else
                            masks[i][j] = null;
                    });
                }
                // calculate output dimension
                var largeBlockMaxDim = (numberOfLargeBlocks == 0) ? 0 : (dim / InputShape[0]) * (1 + Stride[0] * (largeBlockSize - 1)) + blockOffsets[numberOfLargeBlocks - 1];
                var smallBlockMaxDim = (dim / InputShape[0]) * (1 + Stride[0] * (smallBlockSize - 1)) + blockOffsets[blockOffsets.Length - 1];
                outputDim = (largeBlockMaxDim > smallBlockMaxDim) ? largeBlockMaxDim : smallBlockMaxDim;
                HotIndices = Vector<double>.Build.DenseOfIndexed(outputDim, CornersMap.Select(x => new Tuple<int, double>(x, 1)));
                layerPrepared = true;
            }
        }


        public override IMatrix Apply(IMatrix m)
        {
            if (m.ColumnCount != 1) throw new Exception("Expecting only a single column");
            if (!layerPrepared) Prepare();
            var res = new IVector[masks.Length];
            var v = m.GetColumn(0);
            ParallelProcessInEnv(masks.Length, (env, taskID, k) =>
            {
                res[k] = v.Permute(masks[k], shifts[k], (ulong)outputDim, env);
            });
            var mat = Factory.GetMatrix(res, EMatrixFormat.ColumnMajor, CopyVectors: false);
            return mat;
        }

        public override int OutputDimension()
        {
            return outputDim;
        }

        public double[] RearrangeWeights(double[] weights)
        {
            if (!layerPrepared) Prepare();

            int maps = weights.Length / convolutionEngine.Corners.Length; 
            var NewOrder = new double[maps * outputDim]; 
            for (int i = 0; i < maps; i++)
            {
                for (int j = 0; j < convolutionEngine.Corners.Length; j++)
                {
                    NewOrder[i * outputDim + CornersMap[j]] = weights[j + i * convolutionEngine.Corners.Length];
                }
            }
            return NewOrder;
        }
    }
}
