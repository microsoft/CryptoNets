// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System.IO;
using Caltech101;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using HEWrapper;

namespace NeuralNetworksTest
{
    [TestClass]
    public class CaltechTests
    {
        readonly IFactory Factory = Defaults.RawFactory;

        [TestMethod]
        public void IniReaderTest()
        {
            var ini = new IniReader(@"cal.model.ini", 4096, 102);
            var v = Vector<double>.Build.Dense(4096);
            var bias = TestNetwork.Score(v);
            int N = 10;
            for (int b = 0; b < N; b++)
            {
                Assert.AreEqual(bias[b], ini.Bias[b], 1e-5);
            }
            for (int f = 0; f < 1000; f++)
            {
                v[f] = 1;
                var pred = TestNetwork.Score(v);
                v[f] = 0;
                for (int b = 0; b < N; b++)
                {
                    Assert.AreEqual(pred[b], ini.Bias[b] + ini.Weights[b * 4096 + f], 1e-5);
                }
            }
        }

        [TestMethod]
        public void CalReaderTest()
        {
            var FileName = "cal_deep_test.tsv";
            var lines = File.ReadAllLines(FileName);
            var readerLayer = new LLSingleLineReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0,
                Scale = 1E+10,
            };
            var v = readerLayer.GetNext().Decrypt(null).Column(0);
            var sum = v.Sum();
            Assert.AreEqual(1231.43961, sum, 1e-5);
            var sum2 = v.PointwiseMultiply(v).Sum();
            Assert.AreEqual(3372.6, sum2, 1e-2);
        }

        public class DebugLayer : BaseLayer
        {
            public double[] scores;
            public override IMatrix Apply(IMatrix m)
            {
                scores = TestNetwork.Score(m.GetColumn(0).Decrypt(null));
                return m;
            }
        }


        [TestMethod]
        public void CalPrediction()
        {
            var FileName = "cal_deep_test.tsv";
            var ini = new IniReader(@"cal.model.ini", 4096, 102);
            double weightscale = 1e+6;

            var readerLayer = new LLSingleLineReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0,
                Scale = 1E+10,
            };
            var debugLayer = new DebugLayer() { Source = readerLayer };
            var encryptLayer = new EncryptLayer() { Source = debugLayer };
            var denseLayer = new LLDenseLayer()
            {
                Source = encryptLayer,
                Weights = ini.Weights,
                Bias = ini.Bias,
                WeightsScale = weightscale,
                InputFormat = EVectorFormat.dense
            };

            var network = denseLayer;
            network.PrepareNetwork();
            var pred = network.GetNext().GetColumn(0).Decrypt(null);
            Assert.AreEqual(102, pred.Count);
            for (int i = 0; i < 10; i++)
                Assert.AreEqual(debugLayer.scores[i], pred[i], 1e-3);

        }
    }
}
