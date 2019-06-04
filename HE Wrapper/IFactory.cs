// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace HEWrapper
{
    /// <summary>
    /// The factory is used to:
    /// - generate encrypted vectors/matrices
    /// - create encryption parameters
    /// - save/load encryption parameters
    /// </summary>
    public interface IFactory
    {
        /// <summary>
        /// generate a plaintext vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="format">type of vector to generate (sparse/dense)</param>
        /// <param name="scale">scale is the precision of the values. Each value is multiplied by scale beore rounded to an integer</param>
        /// <returns>a vector</returns>
        IVector GetPlainVector(Vector<double> v, EVectorFormat format, double scale);
        /// <summary>
        /// generate a plaintext vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="format">type of vector to generate (sparse/dense)</param>
        /// <returns>a vector</returns>
        IVector GetPlainVector(IEnumerable<BigInteger> v, EVectorFormat format);
        /// <summary>
        /// generate an encrypted vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="format">type of vector to generate (sparse/dense)</param>
        /// <param name="scale">scale is the precision of the values. Each value is multiplied by scale beore rounded to an integer</param>
        /// <returns>a vector</returns>
        IVector GetEncryptedVector(Vector<double> v, EVectorFormat format, double scale);
        /// <summary>
        /// generate an encrypted vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="format">type of vector to generate (sparse/dense)</param>
        /// <returns>a vector</returns>
        IVector GetEncryptedVector(IEnumerable<BigInteger> v, EVectorFormat format);
        /// <summary>
        /// Get an unbounded integer from string
        /// </summary>
        /// <param name="str">input string</param>
        /// <returns>unbounded integer</returns>
        BigInteger GetValueFromString(string str);
        /// <summary>
        /// Convert an unbounded integer to string
        /// </summary>
        /// <param name="value">unbounded integer</param>
        /// <returns>string</returns>
        string GetStringFromValue(BigInteger value);
        /// <summary>
        /// Deep copy of a vector
        /// </summary>
        /// <param name="v">The vector to copy</param>
        /// <returns>A new vector</returns>
        IVector CopyVector(IVector v);
        /// <summary>
        /// Load vector from a stream
        /// </summary>
        /// <param name="str">Stream to load from</param>
        /// <returns>The vector</returns>
        IVector LoadVector(StreamReader str);
        /// <summary>
        /// Generate a matrix with plaintext values
        /// </summary>
        /// <param name="m">The data to use to populate the matrix</param>
        /// <param name="format">Matrix format to use</param>
        /// <param name="scale">scale is the precision of the values. Each value is multiplied by scale beore rounded to an integer</param>
        /// <returns>The matrix</returns>
        IMatrix GetPlainMatrix(Matrix<double> m, EMatrixFormat format, double scale);
        /// <summary>
        /// Generate a matrix with encrypted values
        /// </summary>
        /// <param name="m">The data to use to populate the matrix</param>
        /// <param name="format">Matrix format to use</param>
        /// <param name="scale">scale is the precision of the values. Each value is multiplied by scale beore rounded to an integer</param>
        /// <returns>The matrix</returns>
        IMatrix GetEncryptedMatrix(Matrix<double> m, EMatrixFormat format, double scale);
        /// <summary>
        /// Creates a matrix from a collection of vectors
        /// </summary>
        /// <param name="vectors"> the vectors that will form the rows/columns of the matrix</param>
        /// <param name="format"> specifies whether the matrix is column major or row major</param>
        /// <param name="env"> computational environment</param>
        /// <param name="CopyVectors"> by default the vectors are copied. However, if this variable is set to false thy are passed by reference and therefore should not be disposed.</param>
        /// <returns> a matrix</returns>
        IMatrix GetMatrix(IVector[] vectors, EMatrixFormat format, bool CopyVectors = true);
        /// <summary>
        /// Load matrix from stream
        /// </summary>
        /// <param name="str">Stream to load from</param>
        /// <returns>Loaded matrix</returns>
        IMatrix LoadMatrix(StreamReader str);
        /// <summary>
        /// Return a computational environment with which it is posible to operate on matrices and vectors
        /// </summary>
        /// <returns>The computational environment</returns>
        IComputationEnvironment AllocateComputationEnv();
        /// <summary>
        /// De-allocate a computational environment
        /// </summary>
        /// <param name="env">The computational environment to de-allocate</param>
        void FreeComputationEnv(IComputationEnvironment env);
        /// <summary>
        /// Save the factory to a stream
        /// </summary>
        /// <param name="stream">The stream to save to</param>
        /// <param name="withPrivateKeys">If set to true, the secret keys are saved, otherwise the factory is saved without secret keys</param>
        /// <returns>The stream</returns>
        Stream Save(Stream stream, bool withPrivateKeys = false);
        /// <summary>
        /// Save the factory to a file
        /// </summary>
        /// <param name="FileName">File name to save to</param>
        /// <param name="withPrivateKeys">If set to true, the secret keys are saved, otherwise the factory is saved without secret keys</param>
        void Save(string FileName, bool withPrivateKeys = false);
    }

    public class RawComputationalEnvironment : IComputationEnvironment
    {
        public IFactory ParentFactory { get; set; } = null;
        public ulong[] Primes { get; set; }
    }

    public class RawFactory : IFactory
    {
        IComputationEnvironment Parent;
        readonly ulong BlockSize;
        public ulong[] Primes { get; set; }


        public RawFactory(ulong BlockSize)
        {
            this.BlockSize = BlockSize;
        }
        public IVector GetPlainVector(Vector<double> v, EVectorFormat format, double scale)
        {
            return new RawVector(v, scale, BlockSize) { Format = format };
        }

        public IVector GetPlainVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            return new RawVector(v, BlockSize) { Format = format };

        }

        public IVector GetEncryptedVector(Vector<double> v, EVectorFormat format, double scale)
        {
            return new RawVector(v, scale, BlockSize) { Format = format };
        }

        public IVector GetEncryptedVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            return new RawVector(v, BlockSize) { Format = format };

        }

        public IVector LoadVector(StreamReader str)
        {
            return RawVector.Read(str);
        }


        public IVector CopyVector(IVector v) { return new RawVector((RawVector)v); }
        public IMatrix GetPlainMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            return new RawMatrix(m, scale, format, BlockSize);
        }
        public IMatrix GetEncryptedMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            return new RawMatrix(m, scale, format, BlockSize);
        }
        public IMatrix GetMatrix(IVector[] vectors, EMatrixFormat format, bool CopyVectors = true)
        {
            var scale = vectors[0].Scale;
            var columns = vectors.Select(v => (Vector<double>)(v.Data) / scale);
            var mat = Matrix<double>.Build.DenseOfColumnVectors(columns);
            return new RawMatrix(mat, scale, format, BlockSize);
        }

        public IMatrix LoadMatrix(StreamReader str)
        {
            return RawMatrix.Read(str);
        }

        public IComputationEnvironment AllocateComputationEnv()
        {
            if (Parent == null) Parent = new RawComputationalEnvironment() { ParentFactory = this, Primes = Primes};
            return Parent;
        }

        public void FreeComputationEnv(IComputationEnvironment env)
        {
           
        }

        public Stream Save(Stream stream, bool withPrivateKeys = false)
        {
            using (var sw = new StreamWriter(stream))
            {
                sw.WriteLine("<RawFactory>");
                sw.WriteLine(BlockSize);
                sw.WriteLine("</RawFactory>");
            }
            return stream;
        }

        public void Save(string FileName, bool withPrivateKeys = false)
        {
            using (var stream = new FileStream(FileName, FileMode.Create))
            {
                Save(stream, withPrivateKeys);
            }
        }

        public BigInteger GetValueFromString(string str)
        {
            return BigInteger.Parse(str);
        }

        public string GetStringFromValue(BigInteger value)
        {
            return value.ToString();
        }
    }

    public class EncryptedSealBfvFactory : IFactory
    {
        ConcurrentQueue<EncryptedSealBfvEnvironment> environmentQueue = new ConcurrentQueue<EncryptedSealBfvEnvironment>();
        EncryptedSealBfvEnvironment referenceEnvironment;
        const int DefaultDecompositionBitCount = 10;
        const int DefaultGaloisDecompositionBitCount  = 20;

        public EncryptedSealBfvFactory()
        {
            ulong[] primes = new ulong[] { 40961, 65537, 114689, 147457, 188417 };
            EncryptedSealBfvEnvironment eenv = new EncryptedSealBfvEnvironment() { ParentFactory = this };
            eenv.GenerateEncryptionKeys(primes, 4096, DefaultDecompositionBitCount, DefaultGaloisDecompositionBitCount);
            referenceEnvironment = eenv;
        }

        public EncryptedSealBfvFactory(ulong[] primes, ulong n, int DecompositionBitCount = DefaultDecompositionBitCount, int GaloisDecompositionBitCount = DefaultGaloisDecompositionBitCount, int SmallModulusCount = -1)
        {
            EncryptedSealBfvEnvironment eenv = new EncryptedSealBfvEnvironment() { ParentFactory = this };
            eenv.GenerateEncryptionKeys(primes, n, DecompositionBitCount, GaloisDecompositionBitCount, SmallModulusCount);
            referenceEnvironment = eenv;
        }

        public EncryptedSealBfvFactory(string fileName)
        {
            referenceEnvironment = new EncryptedSealBfvEnvironment(fileName)
            {
                ParentFactory = this
            };
        }

        public EncryptedSealBfvFactory(Stream stream)
        {
            referenceEnvironment = new EncryptedSealBfvEnvironment(stream)
            {
                ParentFactory = this
            };
        }
        public IVector CopyVector(IVector v)
        {
            return Utils.ProcessInEnv((env) => new EncryptedSealBfvVector(v, env), this);
        }
        public IComputationEnvironment AllocateComputationEnv()
        {
            environmentQueue.TryDequeue(out EncryptedSealBfvEnvironment env);
            if (env == null)
                env = new EncryptedSealBfvEnvironment(referenceEnvironment);
            return env;
        }


        public void FreeComputationEnv(IComputationEnvironment env)
        {
            var lenv = env as EncryptedSealBfvEnvironment;
            environmentQueue.Enqueue(lenv);
        }

        public Stream Save(Stream stream, bool withPrivateKeys = false)
        {
            return referenceEnvironment.Save(stream, withPrivateKeys);
        }
        public void Save(string fileName, bool withPrivateKeys = false)
        {
            referenceEnvironment.Save(fileName, withPrivateKeys);

        }


        public IVector LoadVector(StreamReader str)
        {
            return Utils.ProcessInEnv(env =>  EncryptedSealBfvVector.Read(str, env as EncryptedSealBfvEnvironment), this);
        }

        public IMatrix LoadMatrix(StreamReader str)
        {
            return Utils.ProcessInEnv(env => EncryptedSealBfvMatrix.Read(str, env as EncryptedSealBfvEnvironment), this);
        }

        public IVector GetPlainVector(Vector<double> v, EVectorFormat format, double scale)
        {
            return Utils.ProcessInEnv(env => new EncryptedSealBfvVector(v, env, scale, EncryptData: false, Format: format), this);
        }
        public IVector GetPlainVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            return Utils.ProcessInEnv((env) =>
                 new EncryptedSealBfvVector(v, env, EncryptData: false, Format: format), this);
        }

        public IVector GetEncryptedVector(Vector<double> v, EVectorFormat format, double scale)
        {
            return Utils.ProcessInEnv((env) =>
                new EncryptedSealBfvVector(v, env, scale, EncryptData: true, Format: format), this);
        }
        public IVector GetEncryptedVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            return Utils.ProcessInEnv((env) =>
                new EncryptedSealBfvVector(v, env, EncryptData: true, Format: format), this);
        }

        public IMatrix GetPlainMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            IMatrix res = null;
            Utils.ProcessInEnv((env) =>
            {
                EncryptedSealBfvVector[] vecs = (format == EMatrixFormat.ColumnMajor) ?
                    m.EnumerateColumns().Select(v => new EncryptedSealBfvVector(v, env, scale, Format: EVectorFormat.dense, EncryptData: false)).ToArray()
                    : m.EnumerateRows().Select(v => new EncryptedSealBfvVector(v, env, scale, Format: EVectorFormat.dense, EncryptData: false)).ToArray();

                res = new EncryptedSealBfvMatrix(vecs, env) { Format = format };
                foreach (var v in vecs) v.Dispose();
            }, this);
            return res;
        }

        public IMatrix GetEncryptedMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            IMatrix res = null;
            Utils.ProcessInEnv((env) =>
            {
                EncryptedSealBfvVector[] vecs = null;
                if (format == EMatrixFormat.ColumnMajor)
                {
                    vecs = new EncryptedSealBfvVector[m.ColumnCount];
                    Utils.ParallelProcessInEnv(vecs.Length, env, (penv, taskIndex, k) =>
                        {
                            vecs[k] = new EncryptedSealBfvVector(m.Column(k), penv, scale, Format: EVectorFormat.dense, EncryptData: true);
                        });
                }
                else
                {
                    vecs = new EncryptedSealBfvVector[m.RowCount];
                    Utils.ParallelProcessInEnv(vecs.Length, env, (penv, taskIndex, k) =>
                    {
                        vecs[k] = new EncryptedSealBfvVector(m.Row(k), penv, scale, Format: EVectorFormat.dense, EncryptData: true);
                    });

                }
                res = new EncryptedSealBfvMatrix(vecs, env) { Format = format };
                foreach (var v in vecs) v.Dispose();
            }, this);
            return res;
        }

        public IMatrix GetMatrix(IVector[] vectors, EMatrixFormat format, bool CopyVectors = true)
        {
            IMatrix mat = null;
            Utils.ProcessInEnv((env) =>
            {
                mat = new EncryptedSealBfvMatrix(Array.ConvertAll(vectors, v => (EncryptedSealBfvVector)v), env, CopyVectors: CopyVectors)
                {
                    Format = format
                };
            }, this);
            return mat;
        }

        public BigInteger GetValueFromString(string str)
        {
            var f = str.Split(',').Select(x => uint.Parse(x)).ToArray();
            BigInteger v = new BigInteger();
            for (int i = 0; i < f.Length; i++)
                v += referenceEnvironment.preComputedCoefficients[i] * f[i];
            v = v % referenceEnvironment.bigFactor;
            return v;
        }

        public string GetStringFromValue(BigInteger value)
        {
            return string.Join(",", this.referenceEnvironment.Environments.Select(e => value % e.plainmodulusValue));

        }
    }

    /// <summary>
    /// Note that this class is not fully implemented. It is just intended to allow getting and releasing environments
    /// </summary>
    public class AtomicEncryptedFactory : IFactory
    {
        ConcurrentQueue<AtomicSealBfvEncryptedEnvironment> environmentQueue = new ConcurrentQueue<AtomicSealBfvEncryptedEnvironment>();
        internal AtomicSealBfvEncryptedEnvironment ReferenceEnvironment { get; set; }
        public IComputationEnvironment AllocateComputationEnv()
        {
            environmentQueue.TryDequeue(out AtomicSealBfvEncryptedEnvironment env);
            if (env == null)
                env = new AtomicSealBfvEncryptedEnvironment(ReferenceEnvironment);
            return env;
        }


        public void FreeComputationEnv(IComputationEnvironment env)
        {
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            environmentQueue.Enqueue(eenv);
        }


        public IVector CopyVector(IVector v)
        {
            throw new NotImplementedException();
        }


        public IMatrix GetEncryptedMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            throw new NotImplementedException();
        }

        public IVector GetEncryptedVector(Vector<double> v, EVectorFormat format, double scale)
        {
            throw new NotImplementedException();
        }
        public IVector GetPlainVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            throw new NotImplementedException();
        }
        public IVector GetEncryptedVector(IEnumerable<BigInteger> v, EVectorFormat format)
        {
            throw new NotImplementedException();
        }

        public IMatrix GetMatrix(IVector[] vectors, EMatrixFormat format, bool CopyVectors = true)
        {
            throw new NotImplementedException();
        }

        public IMatrix GetPlainMatrix(Matrix<double> m, EMatrixFormat format, double scale)
        {
            throw new NotImplementedException();
        }

        public IVector GetPlainVector(Vector<double> v, EVectorFormat format, double scale)
        {
            throw new NotImplementedException();
        }

        public IMatrix LoadMatrix(StreamReader str)
        {
            throw new NotImplementedException();
        }

        public IVector LoadVector(StreamReader str)
        {
            throw new NotImplementedException();
        }

        public Stream Save(Stream stream, bool withPrivateKeys = false)
        {
            throw new NotImplementedException();
        }

        public void Save(string FileName, bool withPrivateKeys = false)
        {
            throw new NotImplementedException();
        }


        public BigInteger GetValueFromString(string str)
        {
            var v = BigInteger.Parse(str) % this.ReferenceEnvironment.plainmodulusValue;
            return v;
        }

        public string GetStringFromValue(BigInteger value)
        {
            return value.ToString();
        }

    }

}
