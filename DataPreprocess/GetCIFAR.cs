// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using ICSharpCode.SharpZipLib.Tar;
using System.IO.Compression;

namespace DataPreprocess
{

    public class GetCIFAR
    {
        static IEnumerable<string> BytesToString(byte[] bytes)
        {
            int imageSize = 3 * 32 * 32 + 1; // the +1 is due to the label column
            for (int i = 0; i < bytes.Length; i += imageSize)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendFormat("{0}", bytes[i]);
                for (int color = 0; color < 3; color++)
                    for (int y = 0; y < 32; y++)
                        for (int x = 0; x < 32; x++)
                            sb.AppendFormat("\t{0}", bytes[(i + 1) + y + 32 * (x + 32 * color)]);
                yield return sb.ToString();

            }
        }

        public static void Run(string[] args)
        {
            if (!File.Exists("cifar-10-binary.tar.gz"))
            {
                Console.WriteLine("Please download the binary version of the CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz");
                return;
            }
            Console.WriteLine("reading cifar-10-binary.tar.gz");
            using (var sr = File.OpenRead("cifar-10-binary.tar.gz"))
            using (var gz = new GZipStream(sr, CompressionMode.Decompress))
            using (var tar = TarArchive.CreateInputTarArchive(gz))
            {
                Console.WriteLine("extracting tar file");
                tar.ExtractContents(".");
            }
            Console.WriteLine("reading test_batch.bin");
            var bytes = File.ReadAllBytes("cifar-10-batches-bin\\test_batch.bin");
            Console.WriteLine("writing cifar-test.tsv");
            File.WriteAllLines("cifar-test.tsv", BytesToString(bytes));
            Console.WriteLine("done");

        }
    }
}