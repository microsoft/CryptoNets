// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using HEWrapper;
using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetworks
{
    public class LLSingleLineReader : BaseLayer, IInputLayer
    {
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
        bool _sparseFormat = true;
        public bool SparseFormat { get { return _sparseFormat; } set { _sparseFormat = value; } }

        int _labelColumn = 0;
        public int LabelColumn { get { return _labelColumn; } set { _labelColumn = value; } }

        public double Scale { get; set; }


        public LLSingleLineReader()
        {
            Factory = Defaults.RawFactory;
        }
        public override INetwork GetSource()
        {
            return null;
        }

        readonly char[] delim = new char[] { '\t' };


        public override void Prepare()
        {
            if (!layerPrepared)
            {
                layerPrepared = true;
            }
        }

        public override IMatrix Apply(IMatrix m)
        {
            return GetNext();
        }

        public override IMatrix GetNext()
        {
            if (sr.EndOfStream) return null;
            if (!layerPrepared) Prepare();
            string line = sr.ReadLine();
            var f = line.Split(delim);
            Vector<double> features = null;
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

                features = Vector<double>.Build.DenseOfIndexed(dim, valueList);
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
                features = Vector<double>.Build.DenseOfArray(featuresArray);
            }
            var mat = Matrix<double>.Build.DenseOfColumnVectors(new Vector<double>[] { features });
            var m = Defaults.RawFactory.GetPlainMatrix(mat, EMatrixFormat.ColumnMajor, Scale);
            return m;

        }

        public override int OutputDimension()
        {
            return dim;
        }

        public override double GetOutputScale()
        {
            return Scale;
        }
    }
}
