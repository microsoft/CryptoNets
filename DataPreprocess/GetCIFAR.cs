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
            StringBuilder sb = new StringBuilder();
            for (var p = 0; p < bytes.Length; p++)
            {
                if (p % 3073 == 0)
                {
                    if (p > 0) yield return sb.ToString();
                    sb.Clear();
                    sb.AppendFormat("{0}", bytes[p]);
                }
                else
                {
                    sb.AppendFormat("\t{0}", bytes[p]);
                }
            }
            yield return sb.ToString();
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