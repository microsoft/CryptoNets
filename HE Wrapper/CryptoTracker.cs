// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using Microsoft.Research.SEAL;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace HEWrapper
{
    /// <summary>
    /// the CryptoTracker is used to measure the availabel noise budget during the execution.
    /// Budget tests are perfomed automatically when the code is in DEBUG mode and are disabled 
    /// in RELEASE mode.
    /// </summary>
    public static class CryptoTracker
    {
        static bool PerformBudgetTests = true;
        /// <summary>
        /// disable budget tests even in DEBUG mode
        /// </summary>
        public static void DisableBudgetTests() => PerformBudgetTests = false;

        /// <summary>
        /// The lowest noise budget that was measured on any ciphertext so far
        /// </summary>
        public static int MinBudgetSoFar { get; private set; } = Int32.MaxValue;

        /// <summary>
        /// reset the counter for the lowest noise budget seen so far
        /// </summary>
        public static void Reset() { MinBudgetSoFar = Int32.MaxValue; }

        /// <summary>
        /// test the noise budget in a ciphertext and updates the minimal budget seen if needed
        /// </summary>
        /// <param name="cipher"> input ciphertext</param>
        /// <param name="decryptor"> a decryptor object that is used to measure the noise level</param>
        [ConditionalAttribute("DEBUG")]
        public static void TestBudget(Ciphertext cipher, Decryptor decryptor)
        {
            if (!PerformBudgetTests) return;
            int budget = decryptor.InvariantNoiseBudget(cipher);
            if (budget < MinBudgetSoFar)
            {
                MinBudgetSoFar = budget;
                Console.WriteLine("Warning: Current minimal budget {0}", MinBudgetSoFar);
                if (budget == 0) throw new Exception("error budget is zero");
            }
        }

        /// <summary>
        /// test the noise budget in a vector and updates the minimal budget seen if needed
        /// </summary>
        /// <param name="res"> input vector</param>
        /// <param name="decryptor"> a decryptor object that is used to measure the noise level</param>
        [ConditionalAttribute("DEBUG")]
        public static void TestBudget(IVector res, IFactory factory)
        {
            if (!PerformBudgetTests) return;
            Utils.ProcessInEnv(env =>
            {
                if (res is EncryptedSealBfvVector v) CryptoTracker.TestVectorBudget(v, env as EncryptedSealBfvEnvironment);
            }, factory);

        }

        /// <summary>
        /// test the noise budget in a vector and updates the minimal budget seen if needed
        /// </summary>
        /// <param name="res"> input vector</param>
        /// <param name="decryptor"> a decryptor object that is used to measure the noise level</param>
        public static int TestVectorBudget(EncryptedSealBfvVector v, EncryptedSealBfvEnvironment lenv)
        {
            var vectors = v.Data as AtomicSealBfvEncryptedVector[];
            for (int i = 0; i < vectors.Length; i++)
            {
                var ciphers = vectors[i].Data as Ciphertext[];
                foreach (var c in ciphers)
                {
                    TestBudget(c, lenv.Environments[i].decryptor);
                }
            }
            return MinBudgetSoFar;
        }

        /// <summary>
        /// print a matrix only in debug mode
        /// </summary>
        /// <param name="m"> a matrix</param>
        /// <param name="factory"> the factory that was used to create the matrix </param>
        /// <param name="name"> an optional name for the matrix</param>
        [ConditionalAttribute("DEBUG")]
        public static void Show(IMatrix m, IFactory factory, string name = "")
        {
            Matrix<double> dec = null;
            Utils.ProcessInEnv((env) => { dec = m.Decrypt(env); }, factory);
            Console.WriteLine("Matrix {0} size {1}x{2} format {3} max {4:F4}", name, dec.ColumnCount, dec.RowCount, Enum.GetName(m.Format.GetType(), m.Format), dec.Enumerate().Max(x => Math.Abs(x)));
            for (int i = 0; i < Math.Min(3, dec.ColumnCount); i++)
            {
                for (int j = 0; j < Math.Min(3, dec.RowCount); j++)
                    Console.Write("{0:F4}\t", dec[i, j]);
                Console.WriteLine();
            }
        }

        /// <summary>
        /// print a vector only in debug mode
        /// </summary>
        /// <param name="v"> a vector</param>
        /// <param name="factory"> factory that was used to create the vector</param>
        /// <param name="name">optional name of the vector</param>
        /// <param name="showAll"> if set then all the vector is displayed, otherwise just the first 3 elements</param>
        [ConditionalAttribute("DEBUG")]
        public static void Show(IVector v, IFactory factory, string name = "", bool showAll = false)
        {
            Vector<double> dec = null;
            Utils.ProcessInEnv((env) => { dec = v.Decrypt(env); }, factory);
            Console.Write("{0} size {1}", name, dec.Count);
            int last = dec.Count;
            if (!showAll && 3 < last) last = 3;
            for (int i = 0; i < last; i++)
            {
                Console.Write("\t{0:F4}", dec[i]);
            }
            Console.WriteLine("\t||\t{0:F4}\t{1:F4}", dec.Min(), dec.Max());
        }
    }
}
