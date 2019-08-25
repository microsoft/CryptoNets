using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using HEWrapper;
using System.Collections.Generic;
using System;

namespace NeuralNetworksTest
{
    public class DummyLayer : BaseLayer
    {
        public IMatrix Data { get; set; }
        public override IMatrix Apply(IMatrix m)
        {
            return Data;
        }

        public override int OutputDimension()
        {
            return (int)(Data.RowCount * Data.ColumnCount);
        }

        public override double GetOutputScale()
        {
            return Data.Scale;
        }
        public override IMatrix GetNext()
        {
            return Data;
        }
    }


    public class SelectionLayer : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            var mr = m as RawMatrix;
            var orig = mr.Data as Matrix<double>;
            var data = Matrix<double>.Build.Dense(1, 32 * 32 * 3);

            for (int x = 0; x < 8; x++)
                for (int y = 0; y < 8; y++)
                    for (int c = 0; c < 3; c++)
                        data[0, x + 32 * (y + 32 * c)] = orig[0,y+32 * (x +32 *c)];
            var raw = new RawMatrix(data, 1, m.Format, m.BlockSize);
            raw.RegisterScale(m.Scale);
            return raw;

        }
    }

    [TestClass]
    public class CifarTests
    {
        double[] KerasWeightVectorConvert(double[] weights, int[] shape)
        {
            double[] res = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                int outputIndex = 0;
                var p = i;
                for (int j = 0; j < shape.Length; j++ )
                {
                    var mod = p % shape[j];
                    outputIndex *= shape[j];
                    outputIndex += mod;
                    p = (p - mod) / shape[j];
                }
                res[outputIndex] = weights[i];
            }
            return res;
        }

        [TestMethod]
        public void TestCifar()
        {
            var factory = new RawFactory(16 * 1024);
            WeightsReader wr = new WeightsReader("cifar_test_weights.csv", "cifar_test_bias.csv");

            var l = new List<Tuple<int, int, double>>
            {
                new Tuple<int, int, double>(0, 0, 1)
            };
            var m = factory.GetEncryptedMatrix(Matrix<double>.Build.SparseOfIndexed(1, 3 * 32 * 32,  l), EMatrixFormat.ColumnMajor, 1);
            var inp = new DummyLayer()
            {
                Data = m,
                Source = null,
                Factory = factory
            };

            var convWeights0 = KerasWeightVectorConvert((double[])wr.Weights[0], new int[] { 5, 3, 3, 3 });


            var convLayer1 = new PoolLayer()
            {
                Source = inp,
                InputShape = new int[] { 3, 32, 32 },
                KernelShape = new int[] { 3, 3, 3 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 1, 1 },
                MapCount = new int[] { 5, 1, 1 },
                WeightsScale = 256000.0,
                Weights = convWeights0,
                Bias = (double[])wr.Biases[0],
                Factory = factory
            };

            var convWeights1 = KerasWeightVectorConvert((double[])wr.Weights[1], new int[] { 10, 5, 32, 32 });
            var denseLayer2 = new PoolLayer
            {
                Source = convLayer1,
                InputShape = new int[] { 5, 32, 32 },
                KernelShape = new int[] { 5, 32, 32 },
                MapCount = new int[] { 10 },
                Stride = new int[] {5, 32, 32},
                Weights = convWeights1,
                Bias = (double[])wr.Biases[1],
                WeightsScale = 1024.0,
                Factory = factory
            };

            denseLayer2.PrepareNetwork();

            var o = denseLayer2.GetNext();

            var env = factory.AllocateComputationEnv();
            Utils.Show(o, factory);

        }
    }
}
