// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class ConvolutionEngine
    {
// inputs:
        public int[] InputShape { get; set; }
        int[] _kernelShape;
        public int[] KernelShape
        {
            get { return _kernelShape; }
            set
            {
                _kernelShape = value.Select(x => x).ToArray(); // deep copy
                UpdateOffsets();
            }
        }
        public int[] Stride { get; set; }
        public bool[] Padding { get; set; }
        public int[] Upperpadding { get; set; } = null;
        public int[] Lowerpadding { get; set; } = null;
        public int[] MapCount { get; set; } = null;

        // outputs:
        public int[][] Offsets { get; private set; }

        public int[][] Corners { get; private set; }
        int maps;

        bool prepared = false;


        IEnumerable<int[]> OffsetGenerator()
        {
            var offset = KernelShape.Select(x => 0).ToArray();
            bool goodToGo = false;
            do
            {
                yield return offset;
                goodToGo = false;
                for (int i = 0; i < KernelShape.Length; i++)
                {
                    offset[i]++;
                    if (offset[i] < KernelShape[i]) { goodToGo = true; break; }
                    offset[i] = 0;
                }
            } while (goodToGo);
        }

        void UpdateOffsets()
        {
            Offsets = OffsetGenerator().Select(v => (int[])v.Clone()).ToArray();
        }

        IEnumerable<int[]> CornerGenerator()
        {
            int[] min = KernelShape.Select((v, i) => - Lowerpadding[i] - ((Padding[i]) ? -(v / 2) : 0)).ToArray();
            int[] max = KernelShape.Select((v, i) => InputShape[i] + Upperpadding[i] - ((Padding[i]) ? ((v + 1) / 2) : v)).ToArray();
            var offset = min.Select(i => i).ToArray(); // deep copy
            bool goodToGo = false;
            do
            {
                yield return offset;
                goodToGo = false;
                for (int i = KernelShape.Length - 1; i >= 0; i--)
                {
                    offset[i] += Stride[i];
                    if (offset[i] <= max[i]) { goodToGo = true; break; }
                    offset[i] = min[i];
                }
            } while (goodToGo);

        }

        // if the Corner is null, computes the relative offset such that if offset=0 then the location is zero
        // if the Corner is present then computes the absolute offset in the input vector
        public int Location(int[] Corner, int[] offset, int[] shape, int bias = 0)
        {
            if (!prepared) Prepare();
            int index = 0;
            for (int i = 0; i < offset.Length; i++)
            {
                int cord = (Corner != null) ? Corner[i] + offset[i] : offset[i];
                if (cord < 0 || cord >= shape[i]) return -1; // padding
                index *= shape[i];
                index += cord;
            }
            return index + bias;

        }

        public void Prepare()
        {
            if (!prepared)
            {
                if (Upperpadding == null)
                    Upperpadding = new int[InputShape.Length];
                if (Lowerpadding == null)
                    Lowerpadding = new int[InputShape.Length];
                if (Padding == null)
                    Padding = new bool[InputShape.Length];

                maps = (MapCount == null) ? 1 : MapCount.Aggregate(1, (acc, val) => acc * val);


                Corners = CornerGenerator().Select(c => (int[])c.Clone()).ToArray();
                prepared = true;
            }
        }

        public double[] GetDenseBias(double[] bias)
        {
            return Enumerable.Range(0, maps).SelectMany(i => Enumerable.Range(0, Corners.Length).Select(c =>  bias[i])).ToArray();
        }
        public double[] GetDenseWeights(double[] weights)
        {
            if (!prepared) Prepare();
            int rows = maps * Corners.Length;
            int columns = InputShape.Aggregate(1, (acc, val) => acc * val);
            int kernelSize = KernelShape.Aggregate(1, (acc, val) => acc * val);
            Matrix<double> mat = Matrix<double>.Build.Dense(rows, columns);

            for (int m = 0; m < maps; m++)
            {
                for (int i = 0; i < Corners.Length; i++)
                {
                    var c = Corners[i];
                    foreach (var o in Offsets)
                    {
                        var l = Location(c, o, InputShape);
                        if (l < 0) continue;
                        var k = Location(null, o, KernelShape);
                        mat[m * Corners.Length + i, l] = weights[k + m * kernelSize];
                    }
                }
            }
            return mat.ToRowMajorArray();
        }
    }
}
