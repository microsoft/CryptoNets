// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System.IO;
using System;
using System.Linq;
using NeuralNetworks;
using HEWrapper;

namespace Caltech101
{
    class Caltech101
    {


        static void Main(string[] args)
        {
            var ini = new IniReader(@"cal.model.ini", 4096, 102);
            ini.Normalize(@"cal.AffineNormalizer.txt");
            var start = DateTime.Now;
            var Factory = new EncryptedSealBfvFactory(new ulong[] { 4300801 }, 4096, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 2);
            double weightscale = 256;
            string FileName = "cal_deep_test.tsv";
            if (!File.Exists(FileName))
            {
                Console.WriteLine("ERROR: Can't find data file {0}", FileName);
                Console.WriteLine("Please use DataPreprocess to obtain the Caltech-101 dataset");
                return;

            }

            var readerLayer = new LLSingleLineReader()
            {
                FileName = FileName,
                SparseFormat = true,
                NormalizationFactor = 1.0,
                Scale = 256,
            };

            var encryptLayer = new EncryptLayer() { Source = readerLayer, Factory = Factory };
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
            int errs = 0;
            var N = 1020;
            IMatrix m = null;
            Utils.ProcessInEnv(env =>
            {
                for (int i = 0; i < N; i++)
                {
                    Utils.Time("Prediction+Encryption", () => m = network.GetNext());
                    var dec = m.Decrypt(env);
                    m.Dispose();
                    var l = readerLayer.Labels[0];
                    int pred = 0;
                    for (int j = 0; j < 101; j++)
                        if (dec[j, 0] > dec[pred, 0]) pred = j;
                    if (pred != l) errs++;
                    Console.WriteLine("errs {0}/{1} accuracy {2:0.000}% {3} prediction {4} label {5}", errs, i + 1, 100 - (100.0 * errs / (i + 1)), TimingLayer.GetStats(), pred, l);


                }
            }, Factory);
            network.DisposeNetwork();
        }
    }
}
