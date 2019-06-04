// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Caltech101
{
    public class IniReader
    {
        public double[] Weights { get; private set; }
        public double[] Bias { get; private set; }

        public IniReader(string FileName, int numberOfFeatures, int numberOfOutputs)
        {
            var items = new Dictionary<int, double>();
            var lines = File.ReadAllLines(FileName);
            string pattern = @"Class_(?<class>[0-9]*)\+(?<feature>(\(Bias\)|f[0-9]*))\t(?<weight>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)";
            Weights = new double[numberOfFeatures * numberOfOutputs];
            Bias = new double[numberOfOutputs];
            foreach (var l in lines)
            {
                var mc = Regex.Matches(l, pattern);
                if (mc.Count > 0)
                {
                    var weight = Double.Parse(mc[0].Groups["weight"].Value);
                    var clss = int.Parse(mc[0].Groups["class"].Value);
                    var featureString = mc[0].Groups["feature"].Value;
                    int feature = int.MaxValue;
                    if (featureString == "(Bias)")
                        Bias[clss] = weight;
                    else
                    {
                        feature = int.Parse(featureString.Substring(1));
                        Weights[clss * numberOfFeatures + feature] = weight;
                    }

                }
            }
        }

        public void Normalize(double[] factor)
        {
            for (int i = 0, j = 0; i < Weights.Length; i++, j++ )
            {
                if (j >= factor.Length) j = 0;
                Weights[i] *= factor[j];
            }

        }

        string Column(string s, int i)
        {
            var f = s.Split();
            return f[i];
        }

        public void Normalize(string AffineNomrmalizationFileName)
        {
            var lines = File.ReadAllLines(AffineNomrmalizationFileName).Skip(1);
            var factor = lines.Where(x => x.Length > 0).Select(x => Double.Parse(Column(x, 2))).ToArray();
            Normalize(factor);

        }


    }
}
