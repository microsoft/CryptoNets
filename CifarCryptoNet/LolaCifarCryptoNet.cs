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
            var factory = new EncryptedSealBfvFactory(new ulong[] {2148728833,2148794369,2149810177}, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7);

            Console.WriteLine("Encryption keys ready {0}", DateTime.Now);


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
                NormalizationFactor = 1.0,
                Scale = 128.0
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
                WeightsScale = 256.0,
                Weights = (double[])wr.Weights[0],
                Bias = (double[])wr.Biases[0]
            };

            var VectorizeLayer2 = new LLVectorizeLayer() { Source = convLayer1 };

            var activationLayer3 = new SquareActivation() { Source = VectorizeLayer2 };

         

            var convEngine = new ConvolutionEngine()
            {
                InputShape = new int[] { 83, 14, 14 },
                KernelShape = new int[] { 83, 6, 6 },
                Padding = new bool[] { false, false, false },
                Stride = new int[] { 83, 2, 2 },
                MapCount = new int[] { 163, 1, 1 }
            };

            var denseLayer4 = new LLDenseLayer
            {
                Source = activationLayer3,
                WeightsScale = 512.0,
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
                WeightsScale = 1024.0,
                InputFormat = EVectorFormat.dense
            };

            var network = denseLayer6;
            Console.WriteLine("Preparing");
            network.PrepareNetwork();
            var m = network.GetNext();
            Utils.Show(m, factory);
            Console.WriteLine("Max computed value {0} ({1})", RawMatrix.Max, Math.Log(RawMatrix.Max) / Math.Log(2));
        }
    }
}
