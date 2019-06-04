// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using NeuralNetworks;
using HEWrapper;

namespace CryptoNets
{
    public class CryptoNets
    {
        static void Main(string[] args)
        {
            string fileName = "MNIST-28x28-test.txt";
            int batchSize = 8192;
            int numberOfRecords = 10000;
            var Factory = new EncryptedSealBfvFactory(new ulong[] { 549764251649, 549764284417 }, (ulong)batchSize);
            int weightscale = 32;

            var ReaderLayer = new BatchReader
            {
                FileName = fileName,
                SparseFormat = true,
                MaxSlots = batchSize,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0
            };

            var EncryptedLayer = new EncryptLayer() { Source = ReaderLayer, Factory = Factory};

            var StartTimingLayer = new TimingLayer() { Source = EncryptedLayer, StartCounters = new string[] { "Batch-Time" } };

            var ConvLayer1 = new PoolLayer()
            {
                Source = StartTimingLayer,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
                MapCount = new int[] { 5, 1 },
                WeightsScale = weightscale,
                Weights = Weights.Weights_0
            };

            var ActivationLayer2 = new SquareActivation() { Source = ConvLayer1 };

            var DenseLayer3 = new PoolLayer()
            {
                Source = ActivationLayer2,
                InputShape = new int[] { 5 * 13 * 13 },
                KernelShape = new int[] { 5 * 13 * 13 },
                Stride = new int[] { 1000 },
                MapCount = new int[] { 100 },
                Weights = Transpose(Weights.Weights_1, 5 * 13 * 13, 100),
                Bias = Weights.Biases_2,
                WeightsScale = weightscale * weightscale
            };

            var ActivationLayer4 = new SquareActivation() { Source = DenseLayer3 };
            var DenseLayer5 = new PoolLayer()
            {
                Source = ActivationLayer4,
                InputShape = new int[] { 100 },
                KernelShape = new int[] { 100 },
                Stride = new int[] { 1000 },
                MapCount = new int[] { 10 },
                Weights = Weights.Weights_3,
                Bias = Weights.Biases_3,
                WeightsScale = weightscale


            };

            var StopTimingLayer = new TimingLayer() { Source = DenseLayer5, StopCounters = new string[] { "Batch-Time" } };
            var network = StopTimingLayer;
            OperationsCount.Reset();
            Console.WriteLine("Preparing");
            network.PrepareNetwork();
            OperationsCount.Print();
            OperationsCount.Reset();
            for (var p = (INetwork)network; p != null; p = p.Source)
                if (p is BaseLayer b) b.Verbose = true;

            int count = 0;
            int errs = 0;
            while(count < numberOfRecords)
            {
                using (var m = network.GetNext())
                    Utils.ProcessInEnv(env =>
                    {
                        var decrypted = m.Decrypt(env);
                        for (int i = 0; i < decrypted.RowCount; i++)
                        {
                            int pred = 0;
                            for (int j = 1; j < decrypted.ColumnCount; j++)
                            {
                                if (decrypted[i, j] > decrypted[i, pred]) pred = j;
                            }
                            if (pred != ReaderLayer.Labels[i]) errs++;
                            count++;
                            if (count % 100 == 0)
                                Console.WriteLine("errs {0}/{1} accuracy {2:0.000}% prediction {3} label {4}", errs, count, 100 - (100.0 * errs / (count)), pred, ReaderLayer.Labels[i]);
                        }
                        Console.WriteLine("Batch size {0} {1}", batchSize, TimingLayer.GetStats());

                    }, Factory);
            }
            Console.WriteLine("errs {0}/{1} accuracy {2:0.000}%", errs, count, 100 - (100.0 * errs / (count)));
            network.DisposeNetwork();
        }
        public static double[] Transpose(double[] weights, int inputShapeSize = -1, int outputMaps = -1)
        {
            var res = new double[weights.Length];
            for (int i = 0; i < inputShapeSize; i++)
            {
                for (int j = 0; j < outputMaps; j++)
                {
                    res[i + inputShapeSize * j] = weights[outputMaps * i + j];
                }
            }
            return res;
        }
    }
}
