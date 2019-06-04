// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using System.Linq;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using CryptoNets;
using System;

namespace NeuralNetworksTest
{
    [TestClass]
    public class LayersTest
    {

        string MNIST = "MNIST-28x28-test.txt";

        [TestMethod]
        public void SingleLineReader()
        {

            var readerLayer = new BatchReader()
            {
                FileName = MNIST,
                SparseFormat = true,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0,
                MaxSlots = 1

            };

            var singleLineReader = new LLSingleLineReader()
            {
                FileName = MNIST,
                SparseFormat = true,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0,
            };
            var read = readerLayer.GetNext().Decrypt(null);
            var pad = singleLineReader.GetNext().Decrypt(null);

            Assert.AreEqual(28 * 28, pad.RowCount);
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    Assert.AreEqual(read[0, i * 28 + j], pad[i * 28 + j, 0]);


        }

        [TestMethod]
        public void EvenPool()
        {
            var Factory = Defaults.RawFactory;
            var MeanPoolLayer = new PoolLayer()
            {
                Factory = Factory,
                InputShape = new int[] { 3, 4, 4 },
                KernelShape = new int[] { 1, 2, 2 },
                Stride = new int[] { 1, 2, 2 }
            };
            MeanPoolLayer.Prepare();

            var data = new double[1, 48];
            for (int i = 0; i < 48; i++) data[0, i] = i;
            var m = Factory.GetEncryptedMatrix(Matrix<double>.Build.DenseOfArray(data), EMatrixFormat.ColumnMajor, 1);
            Utils.ProcessInEnv(env =>
            {
                var t = MeanPoolLayer.Apply(m);
                var res = t.Decrypt(env);
                Assert.AreEqual(12, res.ColumnCount);
                Assert.AreEqual(1, res.RowCount);
                var expected = new double[] { 2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5, 34.5, 36.5, 42.5, 44.5 };
                for (int i = 0; i < 12; i++)
                    Assert.AreEqual(expected[i], res[0, i]);
                t.Dispose();
                m.Dispose();
            }, Factory);
            MeanPoolLayer.Dispose();
        }

        [TestMethod]
        public void PreConvLayer()
        {
            var Factory = Defaults.RawFactory;
            var preConvLayer = new LLPreConvLayer()
            {
                Factory = Factory,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
            };
            preConvLayer.Prepare();
            var lineLength = 28;
            var inp = Enumerable.Range(0, lineLength * lineLength).Select(x => (double)(x + 1)).ToArray();
            Utils.ProcessInEnv(env =>
            {
                var v = Factory.GetEncryptedVector(Vector<double>.Build.DenseOfArray(inp), EVectorFormat.dense, 1);
                var res = preConvLayer.Apply(Factory.GetMatrix(new IVector[] { v }, EMatrixFormat.ColumnMajor));
                var dec = res.Decrypt(env);
                Assert.AreEqual(196, dec.RowCount);
                Assert.AreEqual(25, dec.ColumnCount);

                var values = new HashSet<int>();
                for (int j = 0; j < dec.RowCount; j++)
                {
                    var val = (int)(dec[j, 0]);
                    if (val != 0)
                    {
                        Assert.IsFalse(values.Contains(val));
                        values.Add(val);
                        var x = (val - 1) / 28;
                        var y = (val - 1) % 28;
                        Assert.AreEqual(0, x % 2);
                        Assert.AreEqual(0, y % 2);
                        Assert.IsTrue(x >= 0);
                        Assert.IsTrue(x < 26);
                        Assert.IsTrue(y >= 0);
                        Assert.IsTrue(y < 26);
                    }
                }
                Assert.AreEqual(13 * 13, values.Count);


                for (int i = 1; i < 25; i++)
                {
                    var dx = i / 5;
                    var dy = i % 5;
                    var delta = dy * 28 + dx;
                    for (int j = 0; j < dec.RowCount; j++)
                    {
                        var val = dec[j, i];
                        var val0 = dec[j, 0];
                        if (val0 == 0) Assert.AreEqual(0, val);
                        else
                        {
                            var y = (int)(val0 - 1) / 28;
                            var x = (int)(val0 - 1) % 28;
                            if (x + dx >= 28) Assert.AreEqual(0, val);
                            else if (y + dy >= 28) Assert.AreEqual(0, val);
                            else Assert.AreEqual(val0 + delta, val);

                        }
                    }
                }
            }, Factory);


        }

        [TestMethod]
        public void PoolLayerAsSparseToDense()
        {
            var Factory = Defaults.RawFactory;
            var v = Vector<double>.Build.DenseOfArray(new double[] { 1, 2, 3 });
            var vec = Factory.GetEncryptedVector(v, EVectorFormat.sparse, 1);
            var m = Factory.GetMatrix(new IVector[] { vec }, EMatrixFormat.ColumnMajor);
            var poolLayer = new LLDenseLayer()
            {
                Factory = Factory,
                Weights = new double[] { 1, 0, 0,
                                             0, 1, 0,
                                             0, 0, 1,
                                             1, 0, 0,
                                             0, 1, 0,
                                             0, 0, 1},
                Bias = new double[] { 0, 0, 0, 0, 0, 0 },
                WeightsScale = 1,
                InputFormat = EVectorFormat.sparse,
                Source = new FakeLayer()
            };
            poolLayer.Prepare();
            var res = poolLayer.Apply(m);
            Utils.ProcessInEnv(env =>
            {
                var dec = res.Decrypt(env);
                Assert.AreEqual(1, dec.ColumnCount);
                Assert.AreEqual(6, dec.RowCount);
                for (int i = 0; i < 6; i++)
                    Assert.AreEqual(1 + (i % 3), dec[i, 0]);
            }, Factory);
        }

        class DummyLayer : BaseLayer
        {
            public override IMatrix Apply(IMatrix m)
            {
                throw new NotImplementedException();
            }

            public override double GetOutputScale()
            {
                return 1.0;
            }
        }

        [TestMethod]
        public void CryptoNetsPoolLayer()
        {
            OperationsCount.Reset();
            int batchSize = 8192;

            var Factory = new EncryptedSealBfvFactory(new ulong[] { 549764251649/*, 549764284417*/ }, (ulong)batchSize);


            int weightscale = 32;
            var DenseLayer3 = new PoolLayer()
            {
                Source = new DummyLayer(),
                InputShape = new int[] { 5 * 13 * 13 },
                KernelShape = new int[] { 5 * 13 * 13 },
                Stride = new int[] { 1000 },
                MapCount = new int[] { 100 },
                Weights = CryptoNets.CryptoNets.Transpose(Weights.Weights_1, 5 * 13 * 13, 100),
                Bias = Weights.Biases_2,
                WeightsScale = weightscale * weightscale,
                Factory = Factory
            };
            DenseLayer3.Prepare();
            var input = Matrix<double>.Build.Dense(8192, 5 * 13 * 13);
            var m = Factory.GetEncryptedMatrix(input, EMatrixFormat.ColumnMajor, 1);
            var start = DateTime.Now;
            var z = DenseLayer3.Apply(m);
            var stop = DateTime.Now;
            var time = (stop - start).TotalMilliseconds;
            Console.WriteLine("time {0}", time);
            OperationsCount.Print();
            z.Dispose();
            m.Dispose();
            DenseLayer3.Dispose();
        }
    }
}
