// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using NeuralNetworks;
using HEWrapper;


namespace CifarCryptoNet
{
    public static class LolaCifarCryptoNet
    {
        public static void Main(string[] args)
        {
            WeightsReader wr = new WeightsReader("CifarWeight.csv", "CifarBias.csv");

            Console.WriteLine("Generating encryption keys {0}", DateTime.Now);
            //var factory = new EncryptedSealBfvFactory(new ulong[] {2148728833,2148794369,2149810177}, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7);
            var factory = new RawFactory(16 * 1024);
            //var factory = new RawFactory(1);
            Console.WriteLine("Encryption keys ready {0}", DateTime.Now);
            int numberOfRecords = 10000;


            string fileName = "cifar-test.tsv";
            var readerLayer = new LLConvReader
            {
                FileName = fileName,
                SparseFormat = false,
                InputShape = new int[] { 3, 32, 32 },
                KernelShape = new int[] { 3, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                NormalizationFactor = 1.0 / 256.0,
                Scale = 8.0
            };


            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = factory };

            var convLayer1 = new LLPoolLayer()
            {
                Source = encryptLayer,
                InputShape = new int[] { 3, 32, 32 },
                KernelShape = new int[] { 3, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                MapCount = new int[] { 83, 1, 1 },
                WeightsScale = 128.0,
                Weights = (double[])wr.Weights[0],
                Bias = (double[])wr.Biases[0]
            };

            var VectorizeLayer2 = new LLVectorizeLayer() { Source = convLayer1 };

            var activationLayer3 = new SquareActivation() { Source = VectorizeLayer2 };



            var convEngine = new ConvolutionEngine()
            {
                InputShape = new int[] { 83, 14, 14 },
                KernelShape = new int[] { 83, 6, 6 },
                Upperpadding = new int[] { 0, 2, 2 },
                Lowerpadding = new int[] { 0, 2, 2 },
                Stride = new int[] { 83, 2, 2 },
                MapCount = new int[] { 163, 1, 1 }
            };

            var denseLayer4 = new LLDenseLayer
            {
                Source = activationLayer3,
                WeightsScale = 768.0,
                Weights = convEngine.GetDenseWeights((double[])wr.Weights[1]),
                Bias = convEngine.GetDenseBias((double[])wr.Biases[1]),
                InputFormat = EVectorFormat.dense,
                ForceDenseFormat = true
            };


            var activationLayer5 = new SquareActivation() { Source = denseLayer4 };

            var denseLayer6 = new LLDenseLayer()
            {
                Source = activationLayer5,
                Weights = (double[])wr.Weights[2],
                Bias = (double[])wr.Biases[2],
                WeightsScale = 512.0,
                InputFormat = EVectorFormat.dense
            };



            var network = denseLayer6;
            Console.WriteLine("Preparing");
            network.PrepareNetwork();
            int count = 0;
            int errs = 0;
            while (count < numberOfRecords)
            {
                using (var m = network.GetNext())
                    Utils.ProcessInEnv(env =>
                    {
                        var decrypted = m.Decrypt(env);
                        int pred = 0;
                        for (int j = 1; j < decrypted.RowCount; j++)
                        {
                            if (decrypted[j, 0] > decrypted[pred, 0]) pred = j;
                        }
                        if (pred != readerLayer.Labels[0]) errs++;
                        count++;
                        if (count % 100 == 0)
                            Console.WriteLine("errs {0}/{1} accuracy {2:0.000}% prediction {3} label {4} bits {5}", errs, count, 100 - (100.0 * errs / (count)), pred, readerLayer.Labels[0], Math.Log(RawMatrix.Max) / Math.Log(2));

                    }, factory);
            }
            Console.WriteLine("errs {0}/{1} accuracy {2:0.000}%", errs, count, 100 - (100.0 * errs / (count)));
            network.DisposeNetwork();
            Console.WriteLine("Max computed value {0} ({1})", RawMatrix.Max, Math.Log(RawMatrix.Max) / Math.Log(2));
        }
    }
}
