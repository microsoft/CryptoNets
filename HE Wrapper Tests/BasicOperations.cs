// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HEWrapper;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;



namespace HE_Wrapper_Tests
{
    [TestClass]
    public class BasicOperations
    {
        double[] values1 = new double[] { -1, 9, 3, 20, 1000,  -6945 };
        Vector<double> vec1 = null;
        IVector enc1 = null;
        readonly double[] values2 = new double[] { 8, -22, 5, 4, 254, -12 };
        readonly Vector<double> vec2 = null;
        readonly IVector enc2 = null;
        readonly IVector plain2 = null;
        readonly double scale = 12;
        readonly double[,] values_m = new double[,] { { 1, -2, 3, -44, 5, 7 }, { 99, 12, -88, 22, 16, 13 } };
        Matrix<double> m = null;
        IMatrix mat = null;
        IFactory Factory = new EncryptedSealBfvFactory();

        public BasicOperations()
        {
            vec1 = Vector<double>.Build.DenseOfArray(values1);
            vec2 = Vector<double>.Build.DenseOfArray(values2);
            m = Matrix<double>.Build.DenseOfArray(values_m);
            enc1 = Factory.GetEncryptedVector(vec1, EVectorFormat.dense, scale);
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
        public void Decrypt()
        {
            Utils.ProcessInEnv(env =>
            {
                var dec = enc1.Decrypt(env);
                Compare(vec1, dec);
            }, Factory);
        }

        [TestMethod]
        public void DecryptMatrix()
        {
            Utils.ProcessInEnv(env =>
            {
                var dec = mat.Decrypt(env);
                Compare(m, dec);
            }, Factory);

        }

        [TestMethod]
        public void MatrixColumn()
        {

            Utils.ProcessInEnv(env =>
            {
                var dec = mat.GetColumn(0).Decrypt(env);
                Compare(m.Column(0), dec);
            }, Factory);

        }

        [TestMethod]
        public void MatrixVectorMultiplication()
        {
            var enc_sparse = Factory.GetEncryptedVector(vec1, EVectorFormat.sparse, scale);
            Utils.ProcessInEnv(env =>
            {
                var dec = mat.Mul(enc_sparse, env).Decrypt(env);
                Compare(m.Multiply(vec1), dec);
            }, Factory);
        }
        [TestMethod]
        public void MatrixVectorMultiplicationPlain()
        {
            Utils.ProcessInEnv(env =>
            {
                var enc_plain = Factory.GetPlainVector(vec1, EVectorFormat.sparse, scale);
                var dec = mat.Mul(enc_plain, env).Decrypt(env);
                Compare(m.Multiply(vec1), dec);
            }, Factory);
        }
        [TestMethod]
        public void Add()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.Add(enc2, env).Decrypt(env);
                Compare(vec1.Add(vec2), res);

                var res_p = enc1.Add(plain2, env).Decrypt(env);
                Compare(vec1.Add(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void ElementMultiply()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.PointwiseMultiply(enc2, env).Decrypt(env);
                Compare(vec1.PointwiseMultiply(vec2), res);

                var res_p = enc1.PointwiseMultiply(plain2, env).Decrypt(env);
                Compare(vec1.PointwiseMultiply(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void DotProduct()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.DotProduct(enc2, env).Decrypt(env);
                Assert.AreEqual(vec1.DotProduct(vec2), res[0]);
                var res_p = enc1.DotProduct(plain2, env).Decrypt(env);
                Assert.AreEqual(vec1.DotProduct(vec2), res_p[0]);
            }, Factory);
        }

        [TestMethod]
        public void Sum()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.SumAllSlots(env).Decrypt(env);
                Assert.AreEqual(vec1.Sum(), res[0]);
            }, Factory);
        }

        [TestMethod]
        public void Subtract()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.Subtract(enc2, env).Decrypt(env);
                Compare(vec1.Subtract(vec2), res);

                var res_p = enc1.Subtract(plain2, env).Decrypt(env);
                Compare(vec1.Subtract(vec2), res_p);

            }, Factory);
        }

        [TestMethod]
        public void Meta()
        {
            Utils.ProcessInEnv(env =>
            {
                Assert.AreEqual(true, enc1.IsEncrypted);
                Assert.AreEqual(false, plain2.IsEncrypted);
                Assert.AreEqual(scale, enc1.Scale);
                var enc2 = Factory.CopyVector(enc1);
                enc2.RegisterScale(20);
                Compare(vec1 * scale / 20, enc2.Decrypt(env));
            }, Factory);
        }

        void TestDuplicate(ulong count)
        {
            Console.WriteLine("start for {0} at {1}", count, DateTime.Now);
            Utils.ProcessInEnv(env =>
            {
                var dup = enc1.Duplicate(count, env);
                Console.WriteLine("Duplicating ended at {0}", DateTime.Now);
                Assert.AreEqual(count * 8, dup.Dim);
                var d = dup.Decrypt(env);
                Assert.AreEqual(count * 8, (ulong)d.Count);
                var exp = new double[8];
                for (int i = 0; i < values1.Length; i++)
                    exp[i] = values1[i];
                for (int i = 0; i < d.Count; i++)
                    Assert.AreEqual(exp[i % 8], d[i]);

            }, Factory);

        }

        [TestMethod]
        public void Duplicate()
        {
            TestDuplicate(4096 / 8);
            TestDuplicate(10);
            TestDuplicate(4096 / 8 - 5);

        }

        [TestMethod]
        public void PackedDotProduct()
        {
            Utils.ProcessInEnv(env =>
            {
                var res = enc1.DotProduct(enc2, 4, env).Decrypt(env);
                double exp = 0;
                for (int i = 0; i < 4; i++) exp += values1[i] * values2[i];
                Assert.AreEqual(exp, res[3]);
            }, Factory);
        }

        [TestMethod]
        public void BigPackedDotProduct()
        {
            var data = (Vector<double>.Build.Random(4096) * 10).PointwiseRound();
            var enc = Factory.GetEncryptedVector(data, EVectorFormat.dense, 1);
            Utils.ProcessInEnv(env =>
            {
                var res = enc.DotProduct(enc, 1024, env).Decrypt(env);

                for (int i = 0; i < 4; i++)
                {
                    double exp = 0;
                    for (int j = 0; j < 1024; j++) exp += data[i * 1024 + j] * data[i * 1024 + j];
                    Assert.AreEqual(exp, res[1024 * i + 1023]);
                }
            }, Factory);
        }

        [TestMethod]
        public void InterleaveRaw()
        {
            var data = new double[][] { new double[] { 1, 0, 0, 2, 0, 0 }, new double[] { 3, 0, 0, 4, 0, 0 } };
            var mat = Matrix<double>.Build.DenseOfColumnArrays(data);
            var raw = new RawFactory(4096);
            var m = raw.GetEncryptedMatrix(mat, EMatrixFormat.ColumnMajor, 10);
            var v = m.Interleave(1, null);
            var res = v.Decrypt(null);
            var exp = Vector<double>.Build.DenseOfArray(new double[] { 1, 3, 0, 2, 4, 0 });
            Compare(exp, res);
        }

        [TestMethod]
        public void Interleave()
        {
            var data = new double[][] { new double[] { 1, 0, 0, 2, 0, 0 }, new double[] { 3, 0, 0, 4, 0, 0 } };
            var mat = Matrix<double>.Build.DenseOfColumnArrays(data);
            Utils.ProcessInEnv(env =>
            {
                var m = Factory.GetEncryptedMatrix(mat, EMatrixFormat.ColumnMajor, 10);
                var v = m.Interleave(1, env);
                var res = v.Decrypt(env);
                var exp = Vector<double>.Build.DenseOfArray(new double[] { 1, 3, 0, 2, 4, 0 });
                Compare(exp, res);
            }, Factory);
        }

        [TestMethod]
        public void InterleaveReverse()
        {
            var data = new double[][] { new double[] { 0, 0, 1, 0, 0, 2 }, new double[] { 0, 0, 3, 0, 0, 4 }, new double[] { 0, 0, 5, 0, 0, 6 } };
            var mat = Matrix<double>.Build.DenseOfColumnArrays(data);
            Utils.ProcessInEnv(env =>
            {
                var m = Factory.GetEncryptedMatrix(mat, EMatrixFormat.ColumnMajor, 10);
                var v = m.Interleave(-1, env);
                var res = v.Decrypt(env);
                var exp = Vector<double>.Build.DenseOfArray(new double[] { 5, 3, 1, 6, 4, 2 });
                Compare(exp, res);
            }, Factory);
        }

        [TestMethod]
        public void SaveLoadKeys()
        {
            if (Factory is EncryptedSealBfvFactory factory)
            {
                var now = DateTime.Now;
                factory.Save("keys.keys");
                factory.Save("keys2.keys", true);
                Console.WriteLine("Saved in {0}", (DateTime.Now - now).TotalSeconds);
                var v = Vector<Double>.Build.Dense(new double[] { 1, 2, 3 });
                IVector vEnc = factory.GetEncryptedVector(v, EVectorFormat.dense, 1);
                var factory2 = new EncryptedSealBfvFactory("keys2.keys");
                Vector<double> w = null;
                Utils.ProcessInEnv(env =>
                {
                    w = vEnc.Decrypt(env);
                }, factory2);
                Compare(v, w);
            }
        }


        [TestMethod]
        public void SaveAndLoadMatrix()
        {
            using (var mem = new MemoryStream())
            {
                using (var sw = new StreamWriter(mem))
                {
                    mat.Write(sw);
                    sw.Flush();
                    mem.Position = 0;
                    IMatrix mat2 = null;
                    using (var sr = new StreamReader(mem))
                        mat2 = Factory.LoadMatrix(sr);
                    Utils.ProcessInEnv(env =>
                    Compare(mat.Decrypt(env), mat2.Decrypt(env)), Factory);
                }
            }

        }

        [TestMethod]
        public void PatrialSumAll()
        {
            var factory = new RawFactory(8192);
            var env = factory.AllocateComputationEnv();
            var v = Vector<double>.Build.Dense(1280);
            v[0] = 1;
            var vec = factory.GetEncryptedVector(v, EVectorFormat.dense, 1);
            var s = vec.DotProduct(vec, 128, env);
            var w = s.Decrypt(env);
            for (int i = 0; i < 1280; i++) Assert.AreEqual((i < 128) ? 1.0 : 0.0, w[i]);
            factory.FreeComputationEnv(env);
        }

        [TestMethod]
        public void Permute()
        {
            var env = Factory.AllocateComputationEnv();
            var values = Enumerable.Range(1, 10).Select(x => (double)x).ToArray();
            var v = Factory.GetEncryptedVector(Vector<double>.Build.DenseOfArray(values), EVectorFormat.dense, 1);
            var I1 = new Tuple<int, double>[] { new Tuple<int, double>(1, 1.0), new Tuple<int, double>(4, 1.0) };
            var I2 = new Tuple<int, double>[] { new Tuple<int, double>(3, 1.0), new Tuple<int, double>(6, 1.0) };
            var S1 = Vector<double>.Build.SparseOfIndexed(10, I1);
            var S2 = Vector<double>.Build.SparseOfIndexed(10, I2);
            var sel1 = Factory.GetPlainVector(S1, EVectorFormat.dense, 1);
            var sel2 = Factory.GetPlainVector(S2, EVectorFormat.dense, 1);
            var w = v.Permute(new IVector[] { sel1, sel2 }, new int[] { 1, 2 }, 5, env);
            var dec = w.Decrypt(env);
            var exp = Vector<double>.Build.DenseOfArray(new double[] { 2, 4, 0, 5, 7 });
            Compare(exp, dec);
        }

        [TestMethod]
        public void BigStack()
        {
            var n = 1050;
            var env = Factory.AllocateComputationEnv();
            var v = new IVector[4];
            for (int i = 0; i < 4; i++)
            {
                var values = Enumerable.Range(i * n, n).Select(x => (double)x).ToArray();
                v[i] = Factory.GetEncryptedVector(Vector<double>.Build.DenseOfArray(values), EVectorFormat.dense, 1);

            }
            var m = Factory.GetMatrix(v, EMatrixFormat.ColumnMajor);
            var vec = m.ConvertToColumnVector(env);
            var dec = vec.Decrypt(env);
            var expArray = Enumerable.Range(0, 4 * n).Select(x => (double)x).ToArray();
            var expVec = Vector<double>.Build.DenseOfArray(expArray);

            Compare(expVec, dec);

        }

        [TestMethod]
        public void GenerateValueFromString()
        {
            var primes = new ulong[] { 40961, 65537, 114689, 147457, 188417 };
            var Factory = new EncryptedSealBfvFactory(primes, 4096);


            var expected = new ulong[] { 21399, 63588, 101610, 90324, 148561 };
            var str = String.Join(",", expected.Select(x => x.ToString()));
            var v = Factory.GetValueFromString(str);
            for (int i = 0; i < primes.Length; i++)
            {
                Assert.AreEqual(expected[i], (ulong)(v % primes[i]));
            }

        }
    }
}
