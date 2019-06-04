// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using MathNet.Numerics.LinearAlgebra;
using HEWrapper;


namespace Basic_Example
{
    class Program
    {
        static void Main(string[] args)
        {
            // without encryption
            //var factory = new RawFactory(4096);
            DateTime start = DateTime.Now;
            // with encryption and allow 60+ bits long numbers
            var Factory = new EncryptedSealBfvFactory();
            //var factory = new EncryptedFactory(new ulong[] { 34359771137}, 16384, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7);
            //var factory = new EncryptedFactory(new ulong[] { 5308417 }, 32768, DecompositionBitCount: 60, GaloisDecompositionBitCount: 60, SmallModulusCount: 7);

            Console.WriteLine("Generated keys in {0} seconds", (DateTime.Now - start).TotalSeconds);
            start = DateTime.Now;
            // with encryption and allow 17+ bits long numbers
            //var factory = new EncryptedFactory(new ulong[] { 188417 }, 4096, 60, 60);
            // with encryption and allow 70+ bits long numbers
            //var factory = new EncryptedFactory(new ulong[] { 40961, 65537, 114689, 147457, 188417 }, 4096);



            Vector<double> v = Vector<double>.Build.DenseOfArray(new double[] { 1, 2, 3 });
            Vector<double> z = Vector<double>.Build.DenseOfArray(new double[] { -1, 5, -4 });

            Utils.ProcessInEnv(env =>
            {
                var chipertext = Factory.GetEncryptedVector(v, EVectorFormat.dense, 1);
                var w = chipertext.DotProduct(chipertext, env);
                var decrypted = w.Decrypt(env);
                Console.WriteLine("Norm Sqared is:\n{0}", decrypted);

                var sum = chipertext.SumAllSlots(env).Decrypt(env);
                Console.WriteLine("sum of elements in a vector:\n{0}", sum);

                var z_chipertext = Factory.GetEncryptedVector(z, EVectorFormat.dense, 1);
                Console.WriteLine("elementwise multiply = \n{0}", chipertext.PointwiseMultiply(z_chipertext, env).Decrypt(env));

            }, Factory);

            Console.WriteLine("Compute in {0} seconds", (DateTime.Now - start).TotalSeconds);

        }
    }
}
