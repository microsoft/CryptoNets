// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Linq;
using NeuralNetworks;
using HEWrapper;
using CommandLine;

namespace LowLatencyCryptoNets
{
    public class LoLaCryptonets
    {
        public enum NetworkStructure {LoLaDense, LoLa, LoLaSmall, LoLaLarge};
        public class Options
        {
            [Option('v', "verbose", Default = false, Required = false, HelpText = "Set output to verbose messages.")]
            public bool Verbose { get; set; }
            [Option('e', "encrypt", Default = false, Required = false, HelpText = "Use encryption")]
            public bool Encrypt { get; set; }
            [Option('n', "network", Required = true, HelpText = "Type of network to use (LoLa, LoLaDense, LoLaSmall, LoLaLarge)")]
            public NetworkStructure NetworkStructure { get; set; }

        }
        static void Main(string[] args)
        {
            var options = new Options();
            var parsed = Parser.Default.ParseArguments<Options>(args).WithParsed(x => options = x);
            if (parsed.Tag == ParserResultType.NotParsed) Environment.Exit(-2);
            var FileName = "MNIST-28x28-test.txt";
            Func<Tuple<INetwork, IFactory>> NetworkCreator = null;
            switch (options.NetworkStructure)
            {
                case NetworkStructure.LoLaDense: NetworkCreator = (() => LoLaDense(FileName, options.Encrypt)); break;
                case NetworkStructure.LoLa:  NetworkCreator = (() => LoLa(FileName, options.Encrypt)); break;
                case NetworkStructure.LoLaSmall: NetworkCreator = (() => SmallLoLa(FileName, options.Encrypt)); break;
                case NetworkStructure.LoLaLarge: NetworkCreator = (() => LargeLoLa(FileName, options.Encrypt)); break;
                default:
                    Console.WriteLine("Unknown command");
                    Environment.Exit(-1);
                    break;
            }
            try
            {
                Evaluate(NetworkCreator, options.Verbose);
            } catch (Exception e)
            {
                Console.WriteLine("Oppsi Daisy!");
#if DEBUG
                Console.WriteLine(BaseLayer.Trace);
#endif
                Console.WriteLine("*****************************************************");
                Console.WriteLine("*****************************************************");
                Console.WriteLine(e.Message);
                Console.WriteLine("*****************************************************");
                Console.WriteLine("*****************************************************");
                Console.WriteLine(e);
            }
            if (options.Verbose && !options.Encrypt)
                Console.WriteLine("Maximal value used {0} ({1:0.00} bits)", RawMatrix.Max, Math.Log(RawMatrix.Max) / Math.Log(2));

        }

        static void Evaluate(Func<Tuple<INetwork, IFactory>> GetNetwork, bool verbose)
        {
            var StartTimingLayer = new TimingLayer() {StartCounters = new string[] { "Prediction-Time" } };
            var StopTimingLayer = new TimingLayer() {StopCounters = new string[] { "Prediction-Time" } };
            IInputLayer ReaderLayer = null;
            var NetworkAndFactory = GetNetwork();
            var Network = NetworkAndFactory.Item1;
            var Factory = NetworkAndFactory.Item2;
            {
                var p = Network;
                while (!(p.GetSource() is EncryptLayer)) p = p.GetSource();

                StartTimingLayer.Source = p.GetSource();
                var b = p as BaseLayer;
                b.Source = StartTimingLayer;

                // find the reader
                while (p.GetSource() != null) p = p.GetSource();
                ReaderLayer = p as IInputLayer;

                // stop the timing counters after computing the entire network
                StopTimingLayer.Source = Network;
                Network = StopTimingLayer;
                p = Network;
                while (p != null)
                {
                    p.Factory = Factory;
                    if (p is BaseLayer bas) bas.Verbose = verbose;
                    p = p.GetSource();
                }

                Network.PrepareNetwork();
            }
            int errs = 0;
            for (int i = 0; i < 10000; i++)
            {
                using (var m = Network.GetNext())
                {
                    var l = ReaderLayer.Labels[0];
                    int pred = 0;
                    Utils.ProcessInEnv(env =>
                    {
                        var dec = m.Decrypt(env);
                        for (int j = 0; j < 10; j++)
                            if (dec[j, 0] > dec[pred, 0]) pred = j;
                        if (pred != l) errs++;
                    }, Factory);

                    Console.WriteLine("errs {0}/{1} accuracy {2:0.000}% {3} prediction {4} label {5}", errs, i + 1, 100 - (100.0 * errs / (i + 1)), TimingLayer.GetStats(), pred, l);
                }
            }
            Network.DisposeNetwork();
        }

        public static Tuple<INetwork, IFactory> LoLaDense(string FileName, bool Encrypt)
        {
            Console.WriteLine("LoLa-Dense mode");
            Console.Write("Generating keys in ");
            var start = DateTime.Now;
            var Factory = (Encrypt) ? (IFactory)new EncryptedSealBfvFactory(new ulong[] { 34359771137, 34360754177 }, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7)
                : (IFactory)new RawFactory(16384);
            var end = DateTime.Now;
            Console.WriteLine("{0} seconds", (end - start).TotalSeconds);

            int weightscale = 32;

            var readerLayer = new LLSingleLineReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0,
            };
            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = Factory };
            var preConvLayer1 = new LLPreConvLayer()
            {
                Source = encryptLayer,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
                UseAxisForBlocks = new bool[] { true, true }
            };
            var ConvLayer2 = new LLPoolLayer()
            {
                Source = preConvLayer1,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
                MapCount = new int[] { 5, 1 },
                WeightsScale = weightscale,
                Weights = Weights.Weights_0,
                HotIndices = preConvLayer1.HotIndices
            };

            var VectorizeLayer3 = new LLVectorizeLayer() { Source = ConvLayer2 };

            var ActivationLayer4 = new SquareActivation() { Source = VectorizeLayer3 };

            var DuplicateLayer5 = new LLDuplicateLayer() { Source = ActivationLayer4, Count = 16};


            var DenseLayer6 = new LLPackedDenseLayer()
            {
                Source = DuplicateLayer5,
                Weights = preConvLayer1.RearrangeWeights(Transpose(Weights.Weights_1, 5 * 13 * 13, 100)),
                Bias = Weights.Biases_2,
                WeightsScale = weightscale * weightscale,
                PackingCount = DuplicateLayer5.Count,
                PackingShift = 1024,
            };

            var ActivationLayer7 = new SquareActivation() { Source = DenseLayer6 };

            var InterleaveLayer8 = new LLInterleaveLayer()
            {
                Source = ActivationLayer7,
                Shift = -1,
                SelectedIndices = Enumerable.Range(0, (int)DuplicateLayer5.Count).Select(i => 1023 + i * 1024).ToList()
            };


            var DenseLayer8 = new LLInterleavedDenseLayer()
            {
                Source = InterleaveLayer8,
                Weights = Weights.Weights_3,
                Bias = Weights.Biases_3,
                WeightsScale = weightscale,
                Shift = -1,
                SelectedIndices = Enumerable.Range(0, (int)DuplicateLayer5.Count).Select(i => 1023 + i * 1024).ToList()

            };

            var network = DenseLayer8;

            return new Tuple<INetwork, IFactory>(network, Factory);
        }

        public static Tuple<INetwork, IFactory> LoLa(string FileName, bool Encrypt)
        {
            Console.WriteLine("LoLa mode");
            Console.Write("Generating keys in ");
            var start = DateTime.Now;
            var Factory = Encrypt ? (IFactory)new EncryptedSealBfvFactory(new ulong[] { 557057, 638977, 737281, 786433 }, 8192) : new RawFactory(8192);
            var end = DateTime.Now;
            Console.WriteLine("{0} seconds", (end - start).TotalSeconds);

            int weightscale = 32;
            
            var readerLayer = new LLConvReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },

            };
            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = Factory };
            var ConvLayer1 = new LLPoolLayer()
            {
                Source = encryptLayer,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
                MapCount = new int[] { 5, 1 },
                WeightsScale = weightscale,
                Weights = Weights.Weights_0
            };
            var VectorizeLayer2 = new LLVectorizeLayer() { Source = ConvLayer1 };

            var ActivationLayer3 = new SquareActivation() { Source = VectorizeLayer2 };

            var DuplicateLayer4 = new LLDuplicateLayer() { Source = ActivationLayer3, Count = 8 };


            var DenseLayer5 = new LLPackedDenseLayer()
            {
                Source = DuplicateLayer4,
                Weights = Transpose(Weights.Weights_1, 5 * 13 * 13, 100),
                Bias = Weights.Biases_2,
                WeightsScale = weightscale * weightscale,
                PackingCount = DuplicateLayer4.Count,
                PackingShift = 1024
            };




            var InterleaveLayer6 = new LLInterleaveLayer()
            {
                Source = DenseLayer5,
                Shift = -1,
                SelectedIndices = Enumerable.Range(0, (int)DuplicateLayer4.Count).Select(i => 1023 + i * 1024).ToList()
            };
            var ActivationLayer7 = new SquareActivation() { Source = InterleaveLayer6 };

            var DenseLayer8 = new LLInterleavedDenseLayer()
            {
                Source = ActivationLayer7,
                Weights = Weights.Weights_3,
                Bias = Weights.Biases_3,
                WeightsScale = weightscale,
                Shift = -1,
                SelectedIndices = Enumerable.Range(0, (int)DuplicateLayer4.Count).Select(i => 1023 + i * 1024).ToList()

            };

            return new Tuple<INetwork, IFactory>(DenseLayer8, Factory);
        }

        public static Tuple<INetwork, IFactory> SmallLoLa(string FileName, bool Encrypt)
        {
            Console.WriteLine("Small LoLa mode");
            Console.Write("Generating keys in ");
            var start = DateTime.Now;
            var Factory = (Encrypt) ? (IFactory)new EncryptedSealBfvFactory(new ulong[] { 2277377, 2424833 }, 8192, DecompositionBitCount: 40, GaloisDecompositionBitCount: 40, SmallModulusCount: 3)
                : new RawFactory(8192);
            var end = DateTime.Now;
            Console.WriteLine("{0} seconds", (end - start).TotalSeconds);

            int weightscale = 64; // with weightscale of 64 the accuracy is 96.92% and the maximal value is 534491448976

            var readerLayer = new LLConvReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0 / 256.0,
                Scale = 16.0,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },

            };
            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = Factory };
            var ConvLayer1 = new LLPoolLayer()
            {
                Source = encryptLayer,
                InputShape = new int[] { 28, 28 },
                KernelShape = new int[] { 5, 5 },
                Upperpadding = new int[] { 1, 1 },
                Stride = new int[] { 2, 2 },
                MapCount = new int[] { 5, 1 },
                WeightsScale = weightscale,
                Weights = SmallModel.Weights_0
            };
            var VectorizeLayer2 = new LLVectorizeLayer() { Source = ConvLayer1 };

            var ActivationLayer3 = new SquareActivation() { Source = VectorizeLayer2 };

            var DenseLayer4 = new LLDenseLayer()
            {
                Source = ActivationLayer3,
                Bias = SmallModel.Biases_1,
                Weights = SmallModel.Weights_1,
                WeightsScale = weightscale,
                InputFormat = EVectorFormat.dense
            };
            return new Tuple<INetwork, IFactory>(DenseLayer4, Factory);
        }


        public static Tuple<INetwork, IFactory> LargeLoLa(string FileName, bool Encrypt)
        {
            Console.WriteLine("Large LoLa mode");
            WeightsReader wr = new WeightsReader("MnistLargeWeight.csv", "MnistLargeBias.csv");
            Console.Write("Generating keys in ");
            var start = DateTime.Now;
            var Factory = Encrypt ? (IFactory)new EncryptedSealBfvFactory(new ulong[] { 2148728833, 2148794369, 2149810177 }, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7)
                : new RawFactory(16384);
            var end = DateTime.Now;
            Console.WriteLine("{0} seconds", (end - start).TotalSeconds);

            var readerLayer = new LLConvReader
            {
                FileName = FileName,
                SparseFormat = true,
                InputShape = new int[] { 1, 28, 28 },
                KernelShape = new int[] { 1, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                NormalizationFactor = 1.0,
                Scale = 16.0
            };


            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = Factory };

            var convLayer1 = new LLPoolLayer()
            {
                Source = encryptLayer,
                InputShape = new int[] { 1, 28, 28 },
                KernelShape = new int[] { 1, 8, 8 },
                Upperpadding = new int[] { 0, 1, 1 },
                Lowerpadding = new int[] { 0, 1, 1 },
                Stride = new int[] { 1000, 2, 2 },
                MapCount = new int[] { 83, 1, 1 },
                WeightsScale = 4096,
                Weights = ((double[])wr.Weights[0]).Select(x => x/256).ToArray(),
                Bias = (double[])wr.Biases[0]
            };

            var VectorizeLayer2 = new LLVectorizeLayer() { Source = convLayer1 };

            var activationLayer3 = new SquareActivation() { Source = VectorizeLayer2 };



            var convEngine = new ConvolutionEngine()
            {
                InputShape = new int[] { 83, 12, 12 },
                KernelShape = new int[] { 83, 6, 6 },
                Padding = new bool[] { false, false, false },
                Stride = new int[] { 83, 2, 2 },
                MapCount = new int[] { 163, 1, 1 }
            };

            var denseLayer4 = new LLDenseLayer
            {
                Source = activationLayer3,
                WeightsScale = 64,
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
                WeightsScale = 512,
                InputFormat = EVectorFormat.dense
            };
            return new Tuple<INetwork, IFactory>(denseLayer6, Factory);
        }



        static double[] Transpose(double[] weights, int inputShapeSize = -1, int outputMaps = -1)
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
