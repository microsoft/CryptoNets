// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using HEWrapper;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class LLConvReader : BaseLayer, IInputLayer
    {
        ConvolutionEngine convolutionEngine = new ConvolutionEngine();
        public int[] InputShape { get { return convolutionEngine.InputShape; } set { convolutionEngine.InputShape = value; layerPrepared = false; } }
        public int[] KernelShape { get { return convolutionEngine.KernelShape; } set { convolutionEngine.KernelShape = value; layerPrepared = false; } }
        public int[] Stride { get { return convolutionEngine.Stride; } set { convolutionEngine.Stride = value; layerPrepared = false; } }
        public bool[] Padding { get { return convolutionEngine.Padding; } set { convolutionEngine.Padding = value; layerPrepared = false; } }
        public int[] Upperpadding { get { return convolutionEngine.Upperpadding; } set { convolutionEngine.Upperpadding = value; layerPrepared = false; } }
        public int[] Lowerpadding { get { return convolutionEngine.Lowerpadding; } set { convolutionEngine.Lowerpadding = value; layerPrepared = false; } }

        int[][] Offsets { get { return convolutionEngine.Offsets; } }
        int[][] Corners { get { return convolutionEngine.Corners; } }



        int dim = -1;

        public double NormalizationFactor { get; set; } = 1.0;
        public int[] Labels { get; private set; }
        string _fileName;


        public string FileName
        {
            get { return _fileName; }
            set
            {
                _fileName = value;
                if (sr != null) sr.Dispose();
                sr = new StreamReader(_fileName);
                dim = -1;
            }
        }
        StreamReader sr = null;

        Vector<double> _features;
        
        /// <summary>
        /// You can set the feature vector and avoid reading from file. The feature vector denotes a single record.
        /// </summary>
        public Vector<double> Features
        {
            get { return _features; }
            set
            {
                _features = value;
                if (_features != null)
                    dim = _features.Count;
            }
        }

        bool _sparseFormat = true;
        public bool SparseFormat { get { return _sparseFormat; } set { _sparseFormat = value; } }

        int _labelColumn = 0;
        public int LabelColumn { get { return _labelColumn; } set { _labelColumn = value; } }

        public double Scale { get; set; }

        public LLConvReader()
        {
            Factory = Defaults.RawFactory;
        }

        public override INetwork GetSource()
        {
            return null;
        }

        public override void Prepare()
        {
            if (!layerPrepared)
            {
                convolutionEngine.Prepare();
                layerPrepared = true;
            }
        }

        readonly char[] Delimiter = new char[] { '\t' };
        public override IMatrix Apply(IMatrix m)
        {
            return GetNext();
        }

        public override IMatrix GetNext()
        {
            if (!layerPrepared)
            {
                convolutionEngine.Prepare();
                layerPrepared = true;
            }
            if (Features == null)//if features are not set, read from file
            {
                if (sr.EndOfStream) return null;
                string line = sr.ReadLine();
                var f = line.Split(Delimiter);
                if (SparseFormat)
                {
                    Labels = new int[] { int.Parse(f[0]) };
                    dim = int.Parse(f[1]);
                    var valueList = new List<Tuple<int, double>>();
                    for (int k = 2; k < f.Length; k++)
                    {
                        string[] sub = f[k].Split(':');
                        int cordinate = int.Parse(sub[0]);
                        double value = double.Parse(sub[1]);
                        valueList.Add(new Tuple<int, double>(cordinate, value * NormalizationFactor));
                    }

                    Features = Vector<double>.Build.DenseOfIndexed(dim, valueList);
                }
                else
                {  //dense format

                    dim = f.Length;
                    if (LabelColumn >= dim)
                        Labels = new int[] { int.MaxValue };
                    else
                    {
                        Labels = new int[] { int.Parse(f[LabelColumn]) };
                        dim--;
                    }
                    double[] featuresArray = new double[dim];

                    for (int k = 0; k < f.Length; k++)
                    {
                        if (k == LabelColumn) continue;
                        featuresArray[(k > LabelColumn) ? k - 1 : k] = double.Parse(f[k]);
                    }
                    Features = Vector<double>.Build.DenseOfArray(featuresArray);
                }
            }
            double[,] mat = new double[Corners.Length, Offsets.Length];
            for (int c = 0; c < Corners.Length; c++)
            {
                for (int o = 0; o < Offsets.Length; o++)
                {
                    var l = convolutionEngine.Location(Corners[c], Offsets[o], InputShape);
                    mat[c, o] = (l >= 0) ? Features[l] : 0; // padding
                }
            }

            var m = new RawMatrix(Matrix<double>.Build.DenseOfArray(mat), Scale, EMatrixFormat.ColumnMajor, 0);
            _features = null;
            return m;

        }

        public override double GetOutputScale()
        {
            return Scale;
        }
    }
}