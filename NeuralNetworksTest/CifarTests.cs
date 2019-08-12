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
        [TestMethod]
        public void TestLargeCifar()
        {
            var factory = new RawFactory(16 * 1024);
            WeightsReader wr = new WeightsReader("large_weights.csv", "large_bias.csv");

            //var l = new List<Tuple<int, int, double>>
            //{
            //    new Tuple<int, int, double>(0, 0, 1),
            //    new Tuple<int, int, double>(0, 1, 1),
            //    new Tuple<int, int, double>(0, 32*32, 1),
            //};
            //var m = factory.GetEncryptedMatrix(Matrix<double>.Build.SparseOfIndexed(1, 3 * 32 * 32, l), EMatrixFormat.ColumnMajor, 1);
            //var inp = new DummyLayer()
            //{
            //    Data = m,
            //    Source = null,
            //    Factory = factory
            //};
            var readerLayer = new BatchReader()
            {
                FileName = "cifar-test.tsv",

                NormalizationFactor = 1.0 / 255.0,
                Scale = 255.0,
                MaxSlots = 1,
                SparseFormat = false
            };

            var selectionLayer = new SelectionLayer()
            {
                Source = readerLayer
            };



            var convLayer1 = new PoolLayer()
            {

                Source = selectionLayer,
                InputShape = new int[] { 3, 32, 32 },
                KernelShape = new int[] { 3, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                MapCount = new int[] { 83, 1, 1 },
                WeightsScale = 2560000.0,
                Weights = (double[])wr.Weights[0],
                Bias = (double[])wr.Biases[0],
                Factory = factory
            };

            var activation2 = new SquareActivation()
            {
                Source = convLayer1,
                Factory = factory
            };

            var convLayer3 = new PoolLayer()
            {
                Source = activation2,
                InputShape = new int[] { 83, 14, 14 },
                KernelShape = new int[] { 83, 6, 6 },
                Upperpadding = new int[] { 0, 2, 2 },
                Lowerpadding = new int[] { 0, 2, 2 },
                Stride = new int[] { 83, 2, 2 },
                MapCount = new int[] { 163, 1, 1 },
                WeightsScale = 2560000.0,
                Weights = (double[])wr.Weights[1],
                Bias = (double[])wr.Biases[1],
                Factory = factory
            };

            var activation4 = new SquareActivation()
            {
                Source = convLayer3,
                Factory = factory
            };

            var convLayer5 = new PoolLayer()
            {
                Source = activation4,
                InputShape = new int[] { 163, 7, 7 },
                KernelShape = new int[] { 163, 7, 7 },
                Stride = new int[] { 1000, 7, 7 },
                MapCount = new int[] { 10, 1, 1 },
                WeightsScale = 2560000.0,
                Weights = (double[])wr.Weights[2],
                Bias = (double[])wr.Biases[2],
                Factory = factory
            };


            var network = convLayer5;
            network.PrepareNetwork();

            var o = network.GetNext();

            var env = factory.AllocateComputationEnv();
            var dec = o.Decrypt(env);
            //var d = Vector<double>.Build.Dense(83);
            //for (int c = 0; c < 83; c++)
            //    d[c] = dec[0, c * 14 * 14 ];
                

        }
    }
}
