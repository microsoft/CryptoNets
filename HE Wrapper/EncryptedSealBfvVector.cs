// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Numerics;
using Microsoft.Research.SEAL;
using System.Threading.Tasks;
using System.IO.Compression;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace HEWrapper
{
    public class EncryptedSealBfvEnvironment : IComputationEnvironment
    {
        AtomicSealBfvEncryptedEnvironment[] _environments;
        public AtomicSealBfvEncryptedEnvironment[] Environments { get { return _environments; } set { _environments = value; PreCompute(); } }
        public BigInteger bigFactor;
        public BigInteger[] preComputedCoefficients;
        public IFactory ParentFactory { get; set; } = null;
        ConcurrentQueue<EncryptedSealBfvEnvironment> environmentQueue = null;

        public EncryptedSealBfvEnvironment() { }
        public EncryptedSealBfvEnvironment(EncryptedSealBfvEnvironment reference)
        {
            bigFactor = reference.bigFactor;
            ParentFactory = reference.ParentFactory;
            preComputedCoefficients = reference.preComputedCoefficients;
            Environments = reference.Environments.Select(x => new AtomicSealBfvEncryptedEnvironment(x)).ToArray();
            if (reference.environmentQueue == null) reference.CreateQueue();
            environmentQueue = reference.environmentQueue;
        }

        void CreateQueue()
        {
            if (environmentQueue != null) return;
            environmentQueue = new ConcurrentQueue<EncryptedSealBfvEnvironment>();
            for (int i = 0; i < Defaults.ThreadCount - 1; i++)
                environmentQueue.Enqueue(new EncryptedSealBfvEnvironment(this));
        }
        public EncryptedSealBfvEnvironment(string fileName)
        {
            using (var stream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                LoadFromStream(stream);
            }
        }

        public EncryptedSealBfvEnvironment(Stream stream)
        {
            LoadFromStream(stream);
        }

        private void LoadFromStream(Stream stream)
        {
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Read, true))
            {
                Environments = archive.Entries.Select(e =>
                {
                    var env = new AtomicSealBfvEncryptedEnvironment();
                    env.ParentFactory = new AtomicEncryptedFactory() { ReferenceEnvironment = env };
                    using (var s = e.Open())
                        env.LoadFromStream(s);
                    return env;
                }).ToArray();
            }
        }

        BigUInt Reminder(BigUInt n1, BigUInt n2, BigUInt res)
        {
            using (var v = n1.DivideRemainder(n2, res))
                return res;

        }

        private void PreCompute()
        {
            var factors = Environments.Select(e => new BigUInt((BigInteger)(e.parameters.PlainModulus.Value)));
            var uIntBigFactor = factors.Aggregate((n1, n2) => n1 * n2);
            bigFactor = (BigInteger)uIntBigFactor.ToBigInteger();
            var minors = factors.Select(p => uIntBigFactor / p).ToList();
            var t = new BigUInt();
            BigUInt[] ys = minors.Zip(factors, (m, p) => Reminder(m, p, t).ModuloInvert(p)).ToArray();
            t.Dispose();
            preComputedCoefficients = ys.Zip(minors, (y, m) => y * m).Select(v => (BigInteger)v.ToBigInteger()).ToArray();
            foreach (var y in ys) t.Dispose();
        }

        public void GenerateEncryptionKeys(ulong[] primes, ulong n, int DecompositionBitCount, int GaloisDecompositionBitCount, int SmallModulusCount = -1)
        {
            var envs = new AtomicSealBfvEncryptedEnvironment[primes.Length];
            Parallel.For(0, primes.Length, i =>
            {
                envs[i] = new AtomicSealBfvEncryptedEnvironment();
                envs[i].GenerateEncryptionKeys(primes[i], n, DecompositionBitCount, GaloisDecompositionBitCount, SmallModulusCount);
                envs[i].ParentFactory = new AtomicEncryptedFactory() { ReferenceEnvironment = envs[i] };
            });
            Environments = envs;
        }

        public Stream Save(Stream stream, bool withPrivateKeys)
        {
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, true))
            {
                var mems = new MemoryStream[Environments.Length];
                Parallel.For(0, mems.Length, i =>
                {
                    mems[i] = new MemoryStream();
                    Environments[i].SaveToStream(mems[i], withPrivateKeys);
                    mems[i].Flush();
                    mems[i].Position = 0;
                });
                for (int i = 0; i < mems.Length; i++)
                {
                    using (var entry = archive.CreateEntry("environment" + i.ToString("D3"), CompressionLevel.NoCompression).Open())
                    {
                        mems[i].CopyTo(entry);
                        mems[i].Dispose();
                    }
                }
            }
            return stream;
        }
        public void Save(string fileName, bool withPrivateKeys)
        {
            using (var fstream = new FileStream(fileName, FileMode.Create))
            {
                Save(fstream, withPrivateKeys);
                fstream.Flush();
            }
        }

        public IComputationEnvironment AllocateComputationEnv()
        {
            if (environmentQueue == null) CreateQueue();
            environmentQueue.TryDequeue(out EncryptedSealBfvEnvironment env);
            return env;
        }

        public void FreeComputationEnv()
        {
            environmentQueue.Enqueue(this);
        }

        public ulong[] Primes { get { return Environments.SelectMany(e => e.Primes).ToArray(); } }
    }
    public class EncryptedSealBfvVector : IVector
    {
        AtomicSealBfvEncryptedVector[] eVectors;
        public object Data { get { return eVectors; } }

        public ulong Dim { get { return (eVectors == null) ? 0 : eVectors[0].Dim; } }

        public double Scale { get; private set; } = 1;

        public ulong BlockSize { get { return eVectors[0].BlockSize; } }

        public bool IsEncrypted { get { return (eVectors == null) ? false : eVectors[0].IsEncrypted; } }
        public EVectorFormat Format { get { return eVectors[0].Format; } }

        public bool IsSigned { get; set; } = true;
#if DEBUG
        readonly string Trace = Environment.StackTrace;
#endif


        public EncryptedSealBfvVector(IVector v, IComputationEnvironment env) // copy constructor
        {
            var lev = v as EncryptedSealBfvVector;
            var lenv = env as EncryptedSealBfvEnvironment;
            Scale = lev.Scale;
            eVectors = lev.eVectors.Zip(lenv.Environments, (x, aenv) => new AtomicSealBfvEncryptedVector(x, aenv)).ToArray();
        }
        public EncryptedSealBfvVector(Vector<double> v, IComputationEnvironment env, double Scale = 1.0, bool EncryptData = true, EVectorFormat Format = EVectorFormat.dense)
        {
            this.Scale = Scale;
            var leenv = env as EncryptedSealBfvEnvironment;
            var splits = SplitBigNumbers(v, leenv);
            eVectors = new AtomicSealBfvEncryptedVector[leenv.Environments.Length];
            Parallel.For(0, eVectors.Length, i =>
            {
                eVectors[i] = new AtomicSealBfvEncryptedVector(splits[i], leenv.Environments[i], Scale: 1, SignedNumbers: false, EncryptData: EncryptData, Format: Format);
            });
        }

        public EncryptedSealBfvVector(IEnumerable<BigInteger> v, IComputationEnvironment env, bool EncryptData = true, EVectorFormat Format = EVectorFormat.dense)
        {
            this.Scale = Scale;
            var leenv = env as EncryptedSealBfvEnvironment;
            var splits = SplitBigNumbers(v, leenv);
            eVectors = new AtomicSealBfvEncryptedVector[leenv.Environments.Length];
            Parallel.For(0, eVectors.Length, i =>
            {
                eVectors[i] = new AtomicSealBfvEncryptedVector(splits[i], leenv.Environments[i], Scale: 1, SignedNumbers: false, EncryptData: EncryptData, Format: Format);
            });
        }

        private EncryptedSealBfvVector() { }

        public EncryptedSealBfvVector(AtomicSealBfvEncryptedVector[] vecs, double Scale = 1.0)
        {
            eVectors = vecs;
            this.Scale = Scale;
        }
#if DEBUG
        ~EncryptedSealBfvVector()
        {
            if (eVectors != null)
                throw new Exception(String.Format("Data that was allocated in the following context was not disposed:\n{0}", Trace));
        }
#endif

        public void Dispose()
        {
            if (eVectors != null)
                foreach (var v in eVectors)
                    if (v != null)
                        v.Dispose();
            eVectors = null;
        }

        AtomicSealBfvEncryptedVector[] ForEveryEncryptedVector(Func<int, Task<AtomicSealBfvEncryptedVector>> lambda)
        {
            var tasks = Enumerable.Range(0, eVectors.Length).Select(i => lambda(i)).ToArray();
            Task.WaitAll(tasks);
            return tasks.Select(t => t.Result).ToArray();
        }
        AtomicSealBfvEncryptedVector[] ForEveryEncryptedVector(Func<int, Task<IVector>> lambda)
        {
            var tasks = Enumerable.Range(0, eVectors.Length).Select(i => lambda(i)).ToArray();
            Task.WaitAll(tasks);
            return tasks.Select(t => (AtomicSealBfvEncryptedVector)t.Result).ToArray();
        }

        static public Task<EncryptedSealBfvVector> InterleaveTask(EncryptedSealBfvVector[] vecs, int shift, EncryptedSealBfvEnvironment env)
        {
            return Task<EncryptedSealBfvVector>.Factory.StartNew(() => Interleave(vecs, shift, env));
        }
        static public EncryptedSealBfvVector Interleave(EncryptedSealBfvVector[] vecs, int shift, EncryptedSealBfvEnvironment env)
        {
            var res = new EncryptedSealBfvVector()
            {
                eVectors = vecs[0].ForEveryEncryptedVector(i => AtomicSealBfvEncryptedVector.InterleaveTask(vecs.Select(v => v.eVectors[i]).ToArray(), shift, env.Environments[i])),
                Scale = vecs[0].Scale
            };
            return res;
        }

        static public Task<EncryptedSealBfvVector> StackTask(EncryptedSealBfvVector[] vecs, EncryptedSealBfvEnvironment env)
        {
            return Task<EncryptedSealBfvVector>.Factory.StartNew(() => Stack(vecs, env));
        }
        static public EncryptedSealBfvVector Stack(EncryptedSealBfvVector[] vecs, EncryptedSealBfvEnvironment env)
        {
            var res = new EncryptedSealBfvVector()
            {
                eVectors = vecs[0].ForEveryEncryptedVector(i => AtomicSealBfvEncryptedVector.StackTask(vecs.Select(v => v.eVectors[i]).ToArray(), env.Environments[i])),
                Scale = vecs[0].Scale
            };
            return res;

        }

        public Task<IVector> AddTask(IVector v, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Add(v, env));
        }
        public IVector Add(IVector v, IComputationEnvironment env)
        {
            if (Scale == 0) return v;
            if (v.Scale == 0) return this;
            if (Scale != v.Scale) throw new Exception("Scales do not match.");
            var leenv = env as EncryptedSealBfvEnvironment;
            var lev = v as EncryptedSealBfvVector;
            var res = new EncryptedSealBfvVector() { Scale = Scale };
            res.eVectors = ForEveryEncryptedVector(i => eVectors[i].AddTask(lev.eVectors[i], leenv.Environments[i]));
            return res;
        }

        public static IVector GenerateSpareOfArray(IVector[] SparseVectors, IComputationEnvironment env)
        {
            var lenv = env as EncryptedSealBfvEnvironment;
            var res = new EncryptedSealBfvVector()
            {
                Scale = SparseVectors[0].Scale,
                eVectors = new AtomicSealBfvEncryptedVector[((EncryptedSealBfvVector)(SparseVectors[0])).eVectors.Length]
            };
            for (int i = 0; i < res.eVectors.Length; i++)
                res.eVectors[i] = AtomicSealBfvEncryptedVector.GenerateSparseOfArray(SparseVectors.Select(v => ((EncryptedSealBfvVector)v).eVectors[i]).ToArray(), lenv.Environments[i]);
            return res;
        }

        static public Task<EncryptedSealBfvVector> DenseMatrixBySparseVectorMultiplyTask(EncryptedSealBfvVector[] denses, EncryptedSealBfvVector sparse, EncryptedSealBfvEnvironment env)
        {
            return Task<EncryptedSealBfvVector>.Factory.StartNew(() => DenseMatrixBySparseVectorMultiply(denses, sparse, env));
        }

        static public EncryptedSealBfvVector DenseMatrixBySparseVectorMultiply(EncryptedSealBfvVector[] denses, EncryptedSealBfvVector sparse, EncryptedSealBfvEnvironment env)
        {
            var res = new EncryptedSealBfvVector()
            {
                Scale = denses[0].Scale * sparse.Scale,
                eVectors = denses[0].ForEveryEncryptedVector(i => AtomicSealBfvEncryptedVector.DenseMatrixBySparseVectorMultiplyTask(denses.Select(d => d.eVectors[i]).ToArray(), sparse.eVectors[i], env.Environments[i]))
            };
            return res;
        }

        public Task<IVector> PointwiseMultiplyTask(IVector v, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => PointwiseMultiply(v, env));
        }
        public IVector PointwiseMultiply(IVector v, IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var lev = v as EncryptedSealBfvVector;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale * v.Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].PointwiseMultiplyTask(lev.eVectors[i], leenv.Environments[i]))
            };
            return res;

        }

        public Task<Vector<double>> DecryptTask(IComputationEnvironment env)
        {
            return Task<Vector<double>>.Factory.StartNew(() => Decrypt(env));
        }
        public Vector<double> Decrypt(IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var splits = eVectors.Zip(leenv.Environments, (v, e) => v.Decrypt(e)).ToArray();
            return JoinSplitNumbers(splits, leenv);
        }

        public Task<IEnumerable<BigInteger>> DecryptFullPrecisionTask(IComputationEnvironment env)
        {
            return Task<IEnumerable<BigInteger>>.Factory.StartNew(() => DecryptFullPrecision(env));
        }
        public IEnumerable<BigInteger> DecryptFullPrecision(IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var splits = eVectors.Zip(leenv.Environments, (v, e) => v.DecryptFullPrecision(e)).ToArray();
            return JoinSplitNumbers(splits, leenv);
        }



        Vector<double>[] SplitBigNumbers(Vector<double> v, EncryptedSealBfvEnvironment leenv)
        {
            var res = new Vector<double>[leenv.Environments.Length];


            var w = v.Multiply(Scale).PointwiseRound().Select(x => (BigInteger)x).ToArray();
            var z = w.Select(x => (x < 0) ? x + leenv.bigFactor : x);

            for (int i = 0; i < res.Length; i++)
            {
                res[i] = Vector<double>.Build.DenseOfEnumerable(z.Select(x => (double)(x % leenv.Environments[i].plainmodulusValue)));
            }
            return res;
        }

        UInt64[][] SplitBigNumbers(IEnumerable<BigInteger> v, EncryptedSealBfvEnvironment leenv)
        {
            var res = new UInt64[leenv.Environments.Length][];


            var z = v.Select(x => (x < 0) ? x + leenv.bigFactor : x);

            for (int i = 0; i < res.Length; i++)
            {
                res[i] = z.Select(x => (UInt64)(x % leenv.Environments[i].plainmodulusValue)).ToArray();
            }
            return res;
        }

        Vector<double> JoinSplitNumbers(Vector<double>[] split, EncryptedSealBfvEnvironment leenv)
        {
            var bigNumbers = new BigInteger[split[0].Count];
            for (int i = 0; i < split.Length; i++)
            {
                for (int j = 0; j < bigNumbers.Length; j++)
                    bigNumbers[j] += ((BigInteger)split[i][j]) * leenv.preComputedCoefficients[i];
            }
            for (int j = 0; j < bigNumbers.Length; j++)
            {
                bigNumbers[j] = bigNumbers[j] % leenv.bigFactor;
                if (bigNumbers[j] * 2 > leenv.bigFactor) bigNumbers[j] = bigNumbers[j] - leenv.bigFactor;
            }
            return Vector<double>.Build.DenseOfEnumerable(bigNumbers.Select(x => (double)x / Scale));
        }

        IEnumerable<BigInteger> JoinSplitNumbers(IEnumerable<BigInteger>[] split, EncryptedSealBfvEnvironment leenv)
        {
            var bigNumbers = new BigInteger[split[0].Count()];
            for (int i = 0; i < split.Length; i++)
            {
                for (int j = 0; j < bigNumbers.Length; j++)
                    bigNumbers[j] += ((BigInteger)split[i].ElementAt(j)) * leenv.preComputedCoefficients[i];
            }
            for (int j = 0; j < bigNumbers.Length; j++)
            {
                bigNumbers[j] = bigNumbers[j] % leenv.bigFactor;
                if (IsSigned && (bigNumbers[j] * 2 > leenv.bigFactor)) bigNumbers[j] = bigNumbers[j] - leenv.bigFactor;
            }
            return bigNumbers;
        }


        public static EncryptedSealBfvVector Read(StreamReader str, EncryptedSealBfvEnvironment env)
        {
            var vct = new EncryptedSealBfvVector();
            var line = str.ReadLine();
            if (line != "<Start LargeEncryptedVector>") throw new Exception("Bad stream format.");
            vct.Scale = double.Parse(str.ReadLine());
            vct.eVectors = new AtomicSealBfvEncryptedVector[int.Parse(str.ReadLine())];
            for (int i = 0; i < vct.eVectors.Length; i++)
            {
                vct.eVectors[i] = AtomicSealBfvEncryptedVector.Read(str, env.Environments[i]);
            }

            line = str.ReadLine();
            if (line != "<End LargeEncryptedVector>") throw new Exception("Bad stream format.");
            return vct;
        }
        public void Write(StreamWriter str)
        {
            str.WriteLine("<Start LargeEncryptedVector>");
            str.WriteLine(Scale);
            str.WriteLine(eVectors.Length);
            for (int i = 0; i < eVectors.Length; i++)
                eVectors[i].Write(str);
            str.WriteLine("<End LargeEncryptedVector>");
            str.Flush();
        }

        public Task<IVector> SubtractTask(IVector v, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Subtract(v, env));
        }
        public IVector Subtract(IVector v, IComputationEnvironment env)
        {
            if (v.Scale == 0) return this;
            if (Scale != v.Scale) throw new Exception("Scales do not match.");

            var leenv = env as EncryptedSealBfvEnvironment;
            var lev = v as EncryptedSealBfvVector;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].SubtractTask(lev.eVectors[i], leenv.Environments[i]))
            };
            return res;
        }

        public Task<IVector> SumAllSlotsTask(ulong length, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => SumAllSlots(length, env));
        }
        public IVector SumAllSlots(ulong length, IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].SumAllSlotsTask(length, leenv.Environments[i]))
            };
            return res;
        }

        public Task<IVector> SumAllSlotsTask(IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => SumAllSlots(env));
        }
        public IVector SumAllSlots(IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var res = new EncryptedSealBfvVector() { Scale = Scale, eVectors = new AtomicSealBfvEncryptedVector[eVectors.Length] };
            Parallel.For(0, eVectors.Length, i =>
              res.eVectors[i] = (AtomicSealBfvEncryptedVector)eVectors[i].SumAllSlots(leenv.Environments[i]));
            return res;
        }


        public Task<IVector> DotProductTask(IVector v, IComputationEnvironment env)
        {
            return ((EncryptedSealBfvVector)PointwiseMultiplyTask(v, env).Result).SumAllSlotsTask(env);
        }
        public IVector DotProduct(IVector v, IComputationEnvironment env) => DotProduct(v, env, null);
        public IVector DotProduct(IVector v, IComputationEnvironment env, int? ForceOutputInColumn = null)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var ev = v as EncryptedSealBfvVector;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale * v.Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].DotProductTask(ev.eVectors[i], leenv.Environments[i], ForceOutputInColumn))
            };
            return res;
        }
        public Task<IVector> DotProductTask(IVector v, ulong length, IComputationEnvironment env)
        {
            return ((EncryptedSealBfvVector)PointwiseMultiplyTask(v, env).Result).SumAllSlotsTask(length, env);
        }

        public IVector DotProduct(IVector v, ulong length, IComputationEnvironment env)
        {
            using (var mul = (EncryptedSealBfvVector)PointwiseMultiply(v, env))
                return mul.SumAllSlots(length, env);
        }
        public void RegisterScale(double scale)
        {
            Scale = scale;
        }

        public Task<IVector> DuplicateTask(ulong count, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Duplicate(count, env));
        }
        public IVector Duplicate(ulong count, IComputationEnvironment env)
        {
            var eenv = env as EncryptedSealBfvEnvironment;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].DuplicateTask(count, eenv.Environments[i]))
            };
            return res;
        }

        public Task<IVector> RotateTask(int amount, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Rotate(amount, env));
        }

        public IVector Rotate(int amount, IComputationEnvironment env)
        {
            var eenv = env as EncryptedSealBfvEnvironment;
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].RotateTask(amount, eenv.Environments[i]))
            };
            return res;
        }

        public Task<IVector> PermuteTask(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Permute(selections, shifts, outputDim, env));
        }

        public IVector Permute(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env)
        {
            var eenv = env as EncryptedSealBfvEnvironment;
            var I = selections.Select((s, i) => (s != null) ? i : -1).Where(i => i >= 0).ToArray();
            var selectI = I.Select(i => selections[i]).ToArray();
            var shiftI = I.Select(i => shifts[i]).ToArray();
            var res = new EncryptedSealBfvVector()
            {
                Scale = Scale,
                eVectors = ForEveryEncryptedVector(i => eVectors[i].PermuteTask(selectI.Select(x => ((EncryptedSealBfvVector)x).eVectors[i]).ToArray(), shiftI, outputDim, eenv.Environments[i]))
            };
            return res;
        }
        internal void RegisterDim(ulong dim)
        {
            foreach (var v in eVectors) v.RegisterDim(dim);
        }
    }
}