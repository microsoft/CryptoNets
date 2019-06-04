// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using HEWrapper;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class BatchReader : BaseLayer, IInputLayer
    {
        int dim = -1;
        int _maxSlots = -1;
        public int MaxSlots { get { return _maxSlots; } set { _maxSlots = value; } }

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

        public override INetwork GetSource()
        {
            return null;
        }

        readonly char[] delim = new char[] { '\t' };

        public BatchReader()
        {
            Factory = Defaults.RawFactory;
        }

        public override IMatrix Apply(IMatrix m)
        {
            return GetNext();
        }

        public override IMatrix GetNext()
        {
            List<int> labelsList = new List<int>();
            List<Vector<Double>> instanceList = new List<Vector<double>>();
            while (!sr.EndOfStream && labelsList.Count < MaxSlots)
            {
                string line = sr.ReadLine();
                var f = line.Split(delim);
                if (SparseFormat)
                {
                    labelsList.Add(int.Parse(f[0]));
                    dim = int.Parse(f[1]);
                    var valueList = new List<Tuple<int, double>>();
                    for (int k = 2; k < f.Length; k++)
                    {
                        string[] sub = f[k].Split(':');
                        int cordinate = int.Parse(sub[0]);
                        double value = double.Parse(sub[1]);
                        valueList.Add(new Tuple<int, double>(cordinate, value * NormalizationFactor));
                    }

                    var features = Vector<double>.Build.DenseOfIndexed(dim, valueList);
                    instanceList.Add(features);
                }
                else
                {  //dense format

                    dim = f.Length;
                    if (LabelColumn >= dim)
                        labelsList.Add(int.MaxValue);
                    else
                    {
                        labelsList.Add(int.Parse(f[LabelColumn]));
                        dim--;
                    }
                    double[] featuresArray = new double[dim];

                    for (int k = 0; k < f.Length; k++)
                    {
                        if (k == LabelColumn) continue;
                        featuresArray[(k > LabelColumn) ? k - 1 : k] = double.Parse(f[k]);
                    }
                    var features = Vector<double>.Build.DenseOfArray(featuresArray);
                    instanceList.Add(features * NormalizationFactor);
                }
            }
            Labels = labelsList.ToArray();
            
            var m = new RawMatrix(Matrix<double>.Build.DenseOfRowVectors(instanceList), Scale, EMatrixFormat.ColumnMajor, 0);
            return m;
        }

        public override double GetOutputScale()
        {
            return Scale;
        }
    }
}