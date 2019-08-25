// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using NeuralNetworks;
using HEWrapper;
using CommandLine;


namespace CifarCryptoNet
{
    public class Options
    {
        [Option('v', "verbose", Default = false, Required = false, HelpText = "Set output to verbose messages.")]
        public bool Verbose { get; set; }
        [Option('e', "encrypt", Default = false, Required = false, HelpText = "Use encryption")]
        public bool Encrypt { get; set; }
    }
    public static class LolaCifarCryptoNet
    {
        public static void Main(string[] args)
        {
            var options = new Options();
            var parsed = Parser.Default.ParseArguments<Options>(args).WithParsed(x => options = x);
            if (parsed.Tag == ParserResultType.NotParsed) Environment.Exit(-2);

            WeightsReader wr = new WeightsReader("CifarWeight.csv", "CifarBias.csv");
            // Model has accuracy of 76.5
            // Current parameters (scale) provide accuracy of 76.31% and uses 78.55 + 1 bits in message length
            // It has a latency of 740 seconds on the reference machine (Azure B8ms server at rest)

            Console.WriteLine("Generating encryption keys {0}", DateTime.Now);
            IFactory factory = null;
            if (options.Encrypt)
                factory = new EncryptedSealBfvFactory(new ulong[] { 957181001729, 957181034497 }, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 8);
            else
                factory = new RawFactory(16 * 1024);
            Console.WriteLine("Encryption keys ready {0}", DateTime.Now);
            int numberOfRecords = 10000;
            bool verbose = options.Verbose;

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
                Scale = 8,
                Verbose = verbose
            };


            var EncryptLayer = new EncryptLayer() { Source = readerLayer, Factory = factory };
            var StartTimingLayer = new TimingLayer() { Source = EncryptLayer, StartCounters = new string[] { "Inference-Time" } };


            var ConvLayer1 = new LLPoolLayer()
            {
                Source = StartTimingLayer,
                InputShape = new int[] { 3, 32, 32 },
                KernelShape = new int[] { 3, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                MapCount = new int[] { 83, 1, 1 },
                WeightsScale = 256.0,
                Weights = (double[])wr.Weights[0],
                Bias = (double[])wr.Biases[0],
                Verbose = verbose
            };

            var VectorizeLayer2 = new LLVectorizeLayer()
            {
                Source = ConvLayer1,
                Verbose = verbose
            };

            var ActivationLayer3 = new SquareActivation()
            {
                Source = VectorizeLayer2,
                Verbose = verbose
            };



            var ConvEngine = new ConvolutionEngine()
            {
                InputShape = new int[] { 83, 14, 14 },
                KernelShape = new int[] { 83, 10, 10 },
                Upperpadding = new int[] { 0, 4, 4 },
                Lowerpadding = new int[] { 0, 4, 4 },
                Stride = new int[] { 83, 2, 2 },
                MapCount = new int[] { 112, 1, 1 }
            };

            var DenseLayer4 = new LLDenseLayer
            {
                Source = ActivationLayer3,
                WeightsScale = 512.0,
                Weights = ConvEngine.GetDenseWeights((double[])wr.Weights[1]),
                Bias = ConvEngine.GetDenseBias((double[])wr.Biases[1]),
                InputFormat = EVectorFormat.dense,
                ForceDenseFormat = true,
                Verbose = verbose
            };


            var ActivationLayer5 = new SquareActivation()
            {
                Source = DenseLayer4,
                Verbose = verbose
            };

            var DenseLayer6 = new LLDenseLayer()
            {
                Source = ActivationLayer5,
                Weights = (double[])wr.Weights[2],
                Bias = (double[])wr.Biases[2],
                WeightsScale = 512.0,
                InputFormat = EVectorFormat.dense,
                Verbose = verbose
            };

            var StopTimingLayer = new TimingLayer() { Source = DenseLayer6, StopCounters = new string[] { "Inference-Time" } };

            var network = StopTimingLayer;
            Console.WriteLine("Preparing");
            network.PrepareNetwork();
            int count = 0;
            int errs = 0;
            int batchSize = 1;
            while (count < numberOfRecords)
            {
                using (var m = network.GetNext())
                {
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
                        if (count % batchSize == 0)
                        {
                            Console.Write("errs {0}/{1} accuracy {2:0.000}% prediction {3} label {4} {5}ms", errs, count, 100 - (100.0 * errs / (count)), pred, readerLayer.Labels[0], TimingLayer.GetStats());
                            if (options.Encrypt)
                                Console.WriteLine();
                            else
                                Console.WriteLine(" {0}bits", Math.Log(RawMatrix.Max) / Math.Log(2));

                        }

                    }, factory);
                }
            }
            Console.WriteLine("errs {0}/{1} accuracy {2:0.000}%", errs, count, 100 - (100.0 * errs / (count)));
            network.DisposeNetwork();
            if (!options.Encrypt)
                Console.WriteLine("Max computed value {0} ({1})", RawMatrix.Max, Math.Log(RawMatrix.Max) / Math.Log(2));
        }
    }
}
