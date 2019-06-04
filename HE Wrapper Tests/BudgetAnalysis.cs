// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HEWrapper;
using MathNet.Numerics.LinearAlgebra;

namespace HE_Wrapper_Tests
{
    [TestClass]
    public class BudgetAnalysis
    {
        readonly double[] values1 = new double[] { -1, 9, 3, 20, 1000, -6945 };
        readonly Vector<double> vec1 = null;
        readonly IVector enc1 = null;
        readonly double[] values2 = new double[] { 8, -22, 5, 4, 254, -12 };
        readonly Vector<double> vec2 = null;
        readonly IVector enc2 = null;
        readonly IVector plain2 = null;
        readonly double scale = 12;
        readonly double[,] values_m = new double[,] { { 1, -2, 3, -44, 5, 7 }, { 99, 12, -88, 22, 16, 13 } };
        readonly Matrix<double> m = null;
        readonly IMatrix mat = null;

        IFactory Factory = null;
        public BudgetAnalysis()
        {
            Factory = new EncryptedSealBfvFactory(new ulong[] { 549764251649/*, 549764284417 */}, 8192);
            vec1 = Vector<double>.Build.DenseOfArray(values1);
            vec2 = Vector<double>.Build.DenseOfArray(values2);
            m = Matrix<double>.Build.DenseOfArray(values_m);
            enc1 = Factory.GetEncryptedVector(vec1, EVectorFormat.dense, scale);
            enc2 = Factory.GetEncryptedVector(vec2, EVectorFormat.dense, scale);
            plain2 = Factory.GetPlainVector(vec2, EVectorFormat.dense, scale);
            mat = Factory.GetEncryptedMatrix(m, EMatrixFormat.ColumnMajor, scale);
        }        [TestMethod]
        public void DotProductBudget()
        {
            Utils.ProcessInEnv(env =>
            {
                CryptoTracker.Reset();
                Console.WriteLine("fresh {0}", CryptoTracker.TestVectorBudget(enc1 as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();

                var res = enc1.DotProduct(enc2, env);
                Console.WriteLine("enc dot product {0}", CryptoTracker.TestVectorBudget(res as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();
                var res_p = enc1.DotProduct(plain2, env);
                Console.WriteLine("plain dot product {0}", CryptoTracker.TestVectorBudget(res_p as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();
                var res_s = enc1.SumAllSlots(env);
                Console.WriteLine("sum slots {0}", CryptoTracker.TestVectorBudget(res_s as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();
                var res_m = enc1.PointwiseMultiply(plain2, env);
                Console.WriteLine("plain multiplication {0}", CryptoTracker.TestVectorBudget(res_m as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();
                var res_em = enc1.PointwiseMultiply(enc2, env);
                Console.WriteLine("enc multiplication {0}", CryptoTracker.TestVectorBudget(res_em as EncryptedSealBfvVector, env as EncryptedSealBfvEnvironment));
                CryptoTracker.Reset();

            }, Factory);
        }


    }
}
