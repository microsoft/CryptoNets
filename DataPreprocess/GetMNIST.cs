// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.IO.Compression;

namespace DataPreprocess
{

    public class GetMNIST
    {

        static byte[] ReadGZFile(string fileName)
        {
            var mem = new MemoryStream();
            var block = new byte[1024];
            using (var sr = File.OpenRead(fileName))
            using (var gz = new GZipStream(sr, CompressionMode.Decompress))
            {
                do
                {
                    var n = gz.Read(block, 0, block.Length);
                    mem.Write(block, 0, n);
                    if (n < block.Length) break;
                } while (true);
            }
            mem.Flush();
            mem.Position = 0;
            var buffer = mem.ToArray();
            mem.Dispose();
            return buffer;
        }

        static string GetImageInSparseFormat(byte[] labels, byte[,] images, int index)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat("{0}\t{1}", labels[index], 28 * 28);
            for (int j = 0; j < 28 * 28; j++)
            {
                if (images[index, j] != 0) sb.AppendFormat("\t{0}:{1}", j, images[index, j]);
            }

            return sb.ToString();
        }

        static IEnumerable<string> GetDatasetInSparseFormat(byte[] labels, byte[,] images)
        {
            for (int i = 0; i < labels.Length; i++)
                yield return GetImageInSparseFormat(labels, images, i);
        }

        public static void Run(string[] args)
        {
            if (!(File.Exists("t10k-images-idx3-ubyte.gz") && File.Exists("t10k-labels-idx1-ubyte.gz")))
            {
                Console.WriteLine("Please download the following files from http://yann.lecun.com/exdb/mnist/");
                Console.WriteLine("\tt10k-images-idx3-ubyte.gz");
                Console.WriteLine("\tt10k-labels-idx1-ubyte.gz");
                return;
            }
            Console.WriteLine("reading input files");
            var imagesBin = ReadGZFile("t10k-images-idx3-ubyte.gz");
            var labelsBin = ReadGZFile("t10k-labels-idx1-ubyte.gz");

            // parse labels
            if (labelsBin[0] != 0 || labelsBin[1] != 0 || labelsBin[2] != 8 || labelsBin[3] != 1)
                throw new Exception("labels file magic number currepted");
            var labels = new byte[labelsBin.Length - 8];
            Buffer.BlockCopy(labelsBin, 8, labels, 0, labels.Length);
            if (imagesBin[0] != 0 || imagesBin[1] != 0 || imagesBin[2] != 8 || imagesBin[3] != 3)
                throw new Exception("images file magic number currepted");
            var images = new byte[labels.Length, 28 * 28];
            Buffer.BlockCopy(imagesBin, 16, images, 0, 28 * 28 * labels.Length);
            Console.WriteLine("writing MNIST-28x28-test.txt");
            File.WriteAllLines("MNIST-28x28-test.txt", GetDatasetInSparseFormat(labels, images));
            Console.WriteLine("done");
        }
    }
}
