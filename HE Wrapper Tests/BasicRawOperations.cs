// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HEWrapper;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace HE_Wrapper_Tests
{
    [TestClass]
    public class RawBasicOperations
    {
        readonly double[] values1 = new double[] { -1, 9, 3, 20, 1000, -6945 };
        Vector<double> vec1 = null;
        IVector enc1 = null;
        readonly double[] values2= new double[] { 8, -22, 5, 4, 254, -12 };
        readonly Vector<double> vec2 = null;
        readonly IVector enc2 = null;
        readonly IVector plain2 = null;
        readonly double scale = 17;
        readonly double[,] values_m = new double[,] { { 1, -2, 3, -44, 5, 7 }, { 99, 12, -88, 22, 16, 13 } };
        Matrix<double> m = null;
        IMatrix mat = null;
        IFactory Factory = Defaults.RawFactory;

        public Vector<double> Vec1 { get => vec1; set => vec1 = value; }
        public RawBasicOperations()
        {
            Vec1 = Vector<double>.Build.DenseOfArray(values1);
            vec2 = Vector<double>.Build.DenseOfArray(values2);
            m = Matrix<double>.Build.DenseOfArray(values_m);
            enc1 = Factory.GetEncryptedVector(Vec1, EVectorFormat.dense, scale);
            enc2 = Factory.GetEncryptedVector(vec2, EVectorFormat.dense, scale);
            plain2 = Factory.GetPlainVector(vec2, EVectorFormat.dense, scale);
            mat = Factory.GetEncryptedMatrix(m, EMatrixFormat.ColumnMajor, scale);
        }

        void Compare(Vector<double> v1, Vector<double> v2)
        {
            Assert.AreEqual(v1.Count, v2.Count);
            for (int i = 0; i < v1.Count; i++)
                Assert.AreEqual(v1[i], v2[i]);
        }

        void Compare(Matrix<double> m1, Matrix<double> m2)
        {
            Assert.AreEqual(m1.RowCount, m2.RowCount);
            Assert.AreEqual(m1.ColumnCount, m2.ColumnCount);
            for (int i = 0; i < m1.RowCount; i++)
                for (int j = 0; j < m1.ColumnCount; j++)
                    Assert.AreEqual(m1[i, j], m2[i, j]);
        }



        [TestMethod]
        public void RawDecrypt()
        {
            Utils.ProcessInEnv(env =>
            {
                var dec = enc1.Decrypt(env);
                Compare(Vec1, dec);
            }, Factory);
        }
        [TestMethod]
        public void RawDecryptMatrix()
        {
            Utils.ProcessInEnv(env =>
            {
                var dec = mat.Decrypt(env);
                Compare(m, dec);
            }, Factory);

        }

        [TestMethod]
        public void RawMatrixColumn()
        {

            Utils.ProcessInEnv(env =>
            {
                var dec = mat.GetColumn(0).Decrypt(env);
                Compare(m.Column(0), dec);
            }, Factory);

        }

        [TestMethod]
        public void RawMatrixVectorMultiplication()
        {
            Utils.ProcessInEnv(env =>
            {
                var enc_sparse = Factory.GetEncryptedVector(Vec1, EVectorFormat.sparse, scale);
                var dec = mat.Mul(enc_sparse, env).Decrypt(env);
                Compare(m.Multiply(Vec1), dec);
            }, Factory);
        }


        [TestMethod]
        public void RawAdd()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.Add(enc2, env).Decrypt(env);
                Compare(Vec1.Add(vec2), res);

                var res_p = enc1.Add(plain2, env).Decrypt(env);
                Compare(Vec1.Add(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void RawElementMultiply()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.PointwiseMultiply(enc2, env).Decrypt(env);
                Compare(Vec1.PointwiseMultiply(vec2), res);

                var res_p = enc1.PointwiseMultiply(plain2, env).Decrypt(env);
                Compare(Vec1.PointwiseMultiply(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void RawDotProduct()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.DotProduct(enc2, env).Decrypt(env);
                Assert.AreEqual(Vec1.DotProduct(vec2), res[0]);
                var res_p = enc1.DotProduct(plain2, env).Decrypt(env);
                Assert.AreEqual(Vec1.DotProduct(vec2), res_p[0]);
            }, Factory);
        }

        [TestMethod]
        public void RawSum()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.SumAllSlots(env).Decrypt(env);
                Assert.AreEqual(Vec1.Sum(), res[0]);
            }, Factory);
        }

        [TestMethod]
        public void RawSubtract()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.Subtract(enc2, env).Decrypt(env);
                Compare(Vec1.Subtract(vec2), res);

                var res_p = enc1.Subtract(plain2, env).Decrypt(env);
                Compare(Vec1.Subtract(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void RawMeta()
        {
            Utils.ProcessInEnv(env =>
            {
                Assert.AreEqual(false, enc1.IsEncrypted);
                Assert.AreEqual(false, plain2.IsEncrypted);
                Assert.AreEqual(scale, enc1.Scale);
                var enc2 = Factory.CopyVector(enc1);
                enc2.RegisterScale(20);
                Compare(Vec1 * scale / 20, enc2.Decrypt(env));
            }, Factory);
        }

        [TestMethod]
        public void RawPermute()
        {
            var factory = new RawFactory(8192);
            var env = factory.AllocateComputationEnv();
            var values = Enumerable.Range(1, 10).Select(x => (double)x).ToArray();
            var v = factory.GetEncryptedVector(Vector<double>.Build.DenseOfArray(values), EVectorFormat.dense, 1);
            var I1 = new Tuple<int, double>[] { new Tuple<int, double>(1, 1.0), new Tuple<int, double>(4, 1.0) };
            var I2 = new Tuple<int, double>[] { new Tuple<int, double>(3, 1.0), new Tuple<int, double>(6, 1.0) };
            var S1 = Vector<double>.Build.SparseOfIndexed(10, I1);
            var S2 = Vector<double>.Build.SparseOfIndexed(10, I2);
            var sel1 = factory.GetPlainVector(S1, EVectorFormat.dense, 1);
            var sel2 = factory.GetPlainVector(S2, EVectorFormat.dense, 1);
            var w = v.Permute(new IVector[] { sel1, sel2 }, new int[] { 1, 2 }, 5, env);
            var dec = w.Decrypt(env);
            var exp = Vector<double>.Build.DenseOfArray(new double[] { 2, 4, 0, 5, 7 });
            Compare(exp, dec);
            factory.FreeComputationEnv(env);
        }

        [TestMethod]
        public void BigStackRaw()
        {
            var factory = new RawFactory(4096);
            var n = 1011;
            var v = new IVector[4];
            for (int i = 0; i < 4; i++)
            {
                var values = Enumerable.Range(i * n, n).Select(x => (double)x).ToArray();
                v[i] = factory.GetEncryptedVector(Vector<double>.Build.DenseOfArray(values), EVectorFormat.dense, 1);

            }
            var m = factory.GetMatrix(v, EMatrixFormat.ColumnMajor);
            Vector<double> dec = null;
            Utils.ProcessInEnv((env) =>
            {
                var vec = m.ConvertToColumnVector(env);
                dec = vec.Decrypt(env);
            }, factory);
            var expArray = Enumerable.Range(0, 4 * n).Select(x => (double)x).ToArray();
            var expVec = Vector<double>.Build.DenseOfArray(expArray);

            Compare(expVec, dec);

        }



    }
}
