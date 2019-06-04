// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.Research.SEAL;
using System.Numerics;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Reflection;
using System.Threading.Tasks;
using System.Threading;

namespace HEWrapper
{
    public class AtomicSealBfvEncryptedEnvironment : IComputationEnvironment
    {

        public Evaluator evaluator;
        public Encryptor encryptor;
        public Decryptor decryptor;
        public EncryptionParameters parameters = null;
        public SEALContext context = null;
        public readonly MemoryPoolHandle memoryPool = MemoryPoolHandle.New();
        public SecretKey secretKey;
        public PublicKey publicKey;
        public RelinKeys relinKeys;
        public BatchEncoder builder;
        public GaloisKeys galoisKeys;
        public Plaintext PlainZero;
        public ulong plainmodulusValue = 0;
        public int PlaintextCapacity { get { return (int)parameters.PolyModulusDegree; } }
        public ulong CiphertextCapacity { get { return 3; } }

        public IFactory ParentFactory { get; set; }


        public AtomicSealBfvEncryptedEnvironment()
        {
        }

        public AtomicSealBfvEncryptedEnvironment(AtomicSealBfvEncryptedEnvironment p)
        {
            parameters = p.parameters;
            context = p.context;
            builder = p.builder;
            relinKeys = p.relinKeys;
            secretKey = p.secretKey;
            publicKey = p.publicKey;
            evaluator = p.evaluator; 
            encryptor = p.encryptor;
            decryptor = p.decryptor;
            galoisKeys = p.galoisKeys;
            PlainZero = p.PlainZero;
            ParentFactory = p.ParentFactory;
            plainmodulusValue = p.plainmodulusValue;
        }

        public void SetKeys(KeyGenerator keys, int DecompositionBitCount, int GaloisDecompositionBitCount)
        {
            evaluator = new Evaluator(context);
            encryptor = new Encryptor(context, keys.PublicKey);
            decryptor = new Decryptor(context, keys.SecretKey);
            builder = new BatchEncoder(context);
            relinKeys = keys.RelinKeys(DecompositionBitCount);
            galoisKeys = keys.GaloisKeys(GaloisDecompositionBitCount);
            secretKey = new SecretKey(keys.SecretKey);
            publicKey = new PublicKey(keys.PublicKey);
            PlainZero = new Plaintext("0", memoryPool);
            plainmodulusValue = parameters.PlainModulus.Value;
        }

        public AtomicSealBfvEncryptedEnvironment GetPublicKeys()
        {
            var publicKeys = new AtomicSealBfvEncryptedEnvironment(this)
            {
                secretKey = null,
                decryptor = null
            };
            return publicKeys;
        }

        public void SaveToFile(string fileName)
        {
            // Open file for output.
            Console.WriteLine("Opening file {0} for writing", fileName);
            using (var file = File.Create(fileName)) SaveToStream(file);
        }

        public void SaveToStream(Stream stream, bool withPrivateKeys = true)
        {
            EncryptionParameters.Save(parameters, stream);
            publicKey.Save(stream);
            relinKeys.Save(stream);
            galoisKeys.Save(stream);
            if (withPrivateKeys)
                secretKey.Save(stream);
            else
                new SecretKey().Save(stream);
        }

        public void LoadFromStream(Stream stream)
        {
            parameters = EncryptionParameters.Load(stream);
            context = SEALContext.Create(parameters);
            publicKey = new PublicKey();
            publicKey.Load(context, stream);
            relinKeys = new RelinKeys();
            relinKeys.Load(context, stream);
            galoisKeys = new GaloisKeys();
            galoisKeys.Load(context, stream);
            secretKey = new SecretKey();
            secretKey.Load(context, stream);
            plainmodulusValue = parameters.PlainModulus.Value;

            evaluator = new Evaluator(context);
            encryptor = new Encryptor(context, publicKey);
            try
            {
                decryptor = new Decryptor(context, secretKey);
            } catch (Exception)
            {
                decryptor = null;
                Console.WriteLine("WARNING: no Secret Key in file. Will not be able to decrypt messages.");
            }
            builder = new BatchEncoder(context);
        }

        public void LoadFromFile(String fileName)
        {
            using (var file = File.OpenRead(fileName))
            {
                LoadFromStream(file);
            }
        }

        public static EncryptionParameters Parms(ulong t, ulong n, int SmallModulusCount)
        {
            var parms = new EncryptionParameters(SchemeType.BFV)
            {
                PlainModulus = new SmallModulus(t),
                PolyModulusDegree = n,
                CoeffModulus = DefaultParams.CoeffModulus128(n)
            };
            if (SmallModulusCount > 0)
                parms.CoeffModulus = parms.CoeffModulus.Take(SmallModulusCount).ToList();
            return parms;
        }
        public static EncryptionParameters Parms(ulong t, ulong n, List<SmallModulus> coefModulus = null)
        {
            var parms = new EncryptionParameters(SchemeType.BFV)
            {
                PlainModulus = new SmallModulus(t),
                PolyModulusDegree = n,
                CoeffModulus = coefModulus
            };
            return parms;
        }

        public void GenerateEncryptionKeys(ulong prime, ulong n, int DecompositionBitCount, int GaloisDecompositionBitCount, int SmallModulusCount)
        {
            GenerateEncryptionKeys(Parms(prime, n, SmallModulusCount), DecompositionBitCount, GaloisDecompositionBitCount);
        }
        public void GenerateEncryptionKeys(EncryptionParameters parms, int DecompositionBitCount, int GaloisDecompositionBitCount)
        {
            this.parameters = parms;
            context = SEALContext.Create(parameters);
            var keyGenerator = new KeyGenerator(context);
            SetKeys(keyGenerator, DecompositionBitCount, GaloisDecompositionBitCount);
        }

        public void GenerateEncryptionKeys(string hexPrime, ulong n, int DecompositionBitCount, int GaloisDecompositionBitCount, int SmallModulusCount = -1)
        {
            GenerateEncryptionKeys(Convert.ToUInt64(hexPrime, 16), n, DecompositionBitCount, GaloisDecompositionBitCount, SmallModulusCount);
        }
        ConcurrentQueue<AtomicSealBfvEncryptedEnvironment> environmentQueue = new ConcurrentQueue<AtomicSealBfvEncryptedEnvironment>();
        public IComputationEnvironment AllocateComputationEnv()
        {
            environmentQueue.TryDequeue(out AtomicSealBfvEncryptedEnvironment env);
            if (env == null)
                env = new AtomicSealBfvEncryptedEnvironment(this);
            return env;

        }


        public void FreeComputationEnv(IComputationEnvironment env)
        {
            environmentQueue.Enqueue(env as AtomicSealBfvEncryptedEnvironment);
        }

        public void Save(string FileName, bool withPrivateKeys)
        {
            throw new NotImplementedException();
        }

        public Stream Save(Stream stream, bool withPrivateKeys)
        {
            throw new NotImplementedException();
        }

        public ulong[] Primes { get { return new ulong[] { plainmodulusValue }; } }
    }

    /// <summary>
    /// This class is used to count the number of operations performed of each type.
    /// </summary>
    public static class OperationsCount
    {
        public static int Destructor, Encryption, Plain, Decryption, Multiplication, PlainMultiplication, Addition, Dispose;
        public static int PlainAddition, Subtraction, PlainSubtraction, Rotation, AddMany, AddManyItemCount, Relinarization;
        static Dictionary<string, int> Totals = null;

        static OperationsCount()
        {
            AppDomain.CurrentDomain.ProcessExit += CurrentDomain_ProcessExit;
        }

        private static void CurrentDomain_ProcessExit(object sender, EventArgs e)
        {
            PrintTotals();
        }
        /// <summary>
        /// report on an operation performed
        /// </summary>
        /// <param name="counter">type of operation performed</param>
        /// <param name="value">number of times the operation was performed</param>
        [ConditionalAttribute("DEBUG")]
        public static void Add(ref int counter, int value)
        {
            Interlocked.Add(ref counter, value);
        }
        /// <summary>
        /// print the number of times each operation was performed
        /// </summary>
        [ConditionalAttribute("DEBUG")]
        public static void Print()
        {
            if (Totals == null) return;
            Type type = typeof(OperationsCount);
            FieldInfo[] fields = type.GetFields();

            Console.WriteLine("Operations:");
            foreach (var f in fields)
            {
                Console.WriteLine("\t{0}\t{1}", f.Name, (int)f.GetValue(f.Name));
            }
        }

        /// <summary>
        /// reset operation counters
        /// </summary>
        [ConditionalAttribute("DEBUG")]
        public static void Reset()
        {
            Type type = typeof(OperationsCount);
            FieldInfo[] fields = type.GetFields();
            if (Totals == null)
                Totals = new Dictionary<string, int>();


            foreach (var f in fields)
            {
                if (!Totals.ContainsKey(f.Name))
                    Totals[f.Name] = (int)f.GetValue(f.Name);
                else
                    Totals[f.Name] += (int)f.GetValue(f.Name);
                f.SetValue(f.Name, 0);
            }
        }

        /// <summary>
        /// print the number of operations performed since the begining of the execution
        /// (numbers are not effected by calls to Reset)
        /// </summary>
        [ConditionalAttribute("DEBUG")]
        public static void PrintTotals()
        {
            if (Totals == null) return;
            Type type = typeof(OperationsCount);
            FieldInfo[] fields = type.GetFields();

            Console.WriteLine("Operations (total):");
            foreach (var f in fields)
            {
                Console.WriteLine("\t{0}\t{1}", f.Name, (int)f.GetValue(f.Name) + Totals[f.Name]);
            }
        }


    }


    /// <summary>
    /// This class represents a vector of plaintext/ciphertext that uses a 
    /// single plaintext modulus and therefore is restricted to hold small numbers.
    /// In most cases this class should not be used directly. Instead use EncryptedVector
    /// which supports large numbers via multiple plaintext moduli
    /// </summary>
    public class AtomicSealBfvEncryptedVector : IVector
    {
        Ciphertext[] encData = null; // if the data is encrypted it will be stored here
        Plaintext[] plainData = null; // if the data is plain it will be stored here

        public bool IsSigned { get; set; } = false;
        public EVectorFormat Format { get; set; } = EVectorFormat.dense;
        public object Data { get { return (encData == null) ? plainData : (object)encData; } }

        /// <summary>
        /// forces the dimension of the vector to the specified number
        /// </summary>
        /// <param name="dim"> new dimension of the vector</param>
        internal void RegisterDim(ulong dim)
        {
            this.Dim = dim;
        }

        public ulong Dim { get; private set; } = 0;

        public ulong BlockSize { get { return (encData != null) ? encData[0].PolyModulusDegree : plainData[0].CoeffCount; } }
        public double Scale { get; private set; }

        public bool IsEncrypted { get { return (plainData == null & encData != null); } }

        /// <summary>
        /// creates a new vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="env">computational environment</param>
        /// <param name="Scale">specifies the precision: the values in v will be multiplied by Scale and rounded to integer</param>
        /// <param name="SignedNumbers"> use signed/unsigned numbers </param>
        /// <param name="EncryptData"> use encrypted/plaintext values</param>
        /// <param name="Format">vector format (sparse/dense)</param>
        public AtomicSealBfvEncryptedVector(Vector<double> v, IComputationEnvironment env, double Scale = 1.0, bool SignedNumbers = true, bool EncryptData = true, EVectorFormat Format = EVectorFormat.dense)
        {
            this.Scale = Scale;
            this.IsSigned = SignedNumbers;
            if (EncryptData) Encrypt(v, Format, env); else Plain(v, Format, env);

        }
        /// <summary>
        /// creates a new vector
        /// </summary>
        /// <param name="v">values to populate the vector with</param>
        /// <param name="env">computational environment</param>
        /// <param name="Scale">specifies the precision: the values in v will be multiplied by Scale and rounded to integer</param>
        /// <param name="SignedNumbers"> use signed/unsigned numbers </param>
        /// <param name="EncryptData"> use encrypted/plaintext values</param>
        /// <param name="Format">vector format (sparse/dense)</param>
        public AtomicSealBfvEncryptedVector(UInt64[] v, IComputationEnvironment env, double Scale = 1.0, bool SignedNumbers = true, bool EncryptData = true, EVectorFormat Format = EVectorFormat.dense)
        {
            this.Scale = Scale;
            this.IsSigned = SignedNumbers;
            if (EncryptData) Encrypt(v, Format, env); else Plain(v, Format, env);

        }
        /// <summary>
        /// copies a vector (deep copy)
        /// </summary>
        /// <param name="v"> vector to be copied</param>
        /// <param name="env"> computational environment</param>
        public AtomicSealBfvEncryptedVector(IVector v, AtomicSealBfvEncryptedEnvironment env) // copy constructor
        {
            var ev = v as AtomicSealBfvEncryptedVector;
            Scale = ev.Scale;
            Dim = ev.Dim;
            IsSigned = ev.IsSigned;
            Format = ev.Format;
            encData = ev.encData?.Select(x => CopyCiphertext(x, env)).ToArray();
            plainData = ev.plainData?.Select(x => { var p = new Plaintext(env.memoryPool); p.Set(x); return p; }).ToArray();
        }


        private AtomicSealBfvEncryptedVector() { }

        ~AtomicSealBfvEncryptedVector()
        {
            OperationsCount.Add(ref OperationsCount.Destructor, 1);
            FreeResources();
        }

        /// <summary>
        /// disposes allocated memory
        /// </summary>
        /// <param name="InDispose"></param>
        void FreeResources()
        {
            if (encData != null)
                foreach (var e in encData) e.Dispose();
            if (plainData != null)
                foreach (var p in plainData) p.Dispose();
            encData = null;
            plainData = null;
        }
        public void Dispose()
        {
            FreeResources();
            GC.SuppressFinalize(this);
            OperationsCount.Add(ref OperationsCount.Dispose, 1);

        }


        /// <summary>
        /// The current version of SEAL does not have a copy constructor for ciphertext so this is a replacement
        /// </summary>
        /// <param name="c"> ciphertext to be copied</param>
        /// <param name="env"> computational environment</param>
        /// <returns></returns>
        private static Ciphertext CopyCiphertext(Ciphertext c, AtomicSealBfvEncryptedEnvironment env) // since there is no copy constructor that takes a memory-pool we have to create one
        {
            var t = new Ciphertext(env.context, c.ParmsId, env.CiphertextCapacity, env.memoryPool);
            t.Set(c);
            return t;
        }

        /// <summary>
        /// allocates 
        /// </summary>
        /// <param name="env"></param>
        /// <returns></returns>
        private static Ciphertext AllocateCiphertext(AtomicSealBfvEncryptedEnvironment env)
        {
            return new Ciphertext(env.context, env.memoryPool);
        }

        static internal Task<AtomicSealBfvEncryptedVector> DenseMatrixBySparseVectorMultiplyTask(AtomicSealBfvEncryptedVector[] denses, AtomicSealBfvEncryptedVector sparse, AtomicSealBfvEncryptedEnvironment env)
        {
            return Task<AtomicSealBfvEncryptedVector>.Factory.StartNew(() => DenseMatrixBySparseVectorMultiply(denses, sparse, env));
        }
        static internal AtomicSealBfvEncryptedVector DenseMatrixBySparseVectorMultiply(AtomicSealBfvEncryptedVector[] denses, AtomicSealBfvEncryptedVector sparse, AtomicSealBfvEncryptedEnvironment env)
        {
            if ((ulong)denses.Length != sparse.Dim) throw new Exception("dimensions do not match");
            if (sparse.Format != EVectorFormat.sparse) throw new Exception("expecting a sparse vector");
            if (!denses[0].IsEncrypted && !sparse.IsEncrypted) throw new Exception("at least one parameter has to be encrypted");
            if (denses[0].IsSigned != sparse.IsSigned) throw new Exception("can't mix signed and unsigned messages");
            if (!denses[0].IsEncrypted && !sparse.IsEncrypted) throw new Exception("at least one parameter has to be encrypted");
            int l = (denses[0].encData != null) ? denses[0].encData.Length : denses[0].plainData.Length;
            bool densesEncrypted = denses[0].IsEncrypted;
            bool sparseEncrypted = sparse.IsEncrypted;
            bool bothEncrypted = densesEncrypted && sparseEncrypted;
            var step1tmp = (bothEncrypted) ? new Ciphertext[Defaults.ThreadCount] : null;
            var step2tmp = new Ciphertext[Defaults.ThreadCount];
            var tmpSum = new Ciphertext[Defaults.ThreadCount][];
            Utils.ParallelProcessInEnv(denses.Length, env, (penv, taskIndex, k) =>
            {
                var epenv = penv as AtomicSealBfvEncryptedEnvironment;
                if (step2tmp[taskIndex] == null)
                {
                    if (bothEncrypted) step1tmp[taskIndex] = AllocateCiphertext(epenv);
                    step2tmp[taskIndex] = AllocateCiphertext(epenv);
                    tmpSum[taskIndex] = new Ciphertext[l];
                }
                for (int i = 0; i < l; i++)
                {
                    if (bothEncrypted)
                    {
                        epenv.evaluator.Multiply(denses[k].encData[i], sparse.encData[k], step1tmp[taskIndex], epenv.memoryPool);
                        epenv.evaluator.Relinearize(step1tmp[taskIndex], epenv.relinKeys, step2tmp[taskIndex], epenv.memoryPool);
                        OperationsCount.Add(ref OperationsCount.Multiplication, 1);
                        OperationsCount.Add(ref OperationsCount.Relinarization, 1);
                    }
                    else if (densesEncrypted)
                    {
                        if (sparse.plainData[k].IsZero)
                            continue;
                        else
                        {
                            epenv.evaluator.MultiplyPlain(denses[k].encData[i], sparse.plainData[k], step2tmp[taskIndex], epenv.memoryPool);
                            OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                        }
                    }
                    else
                    {
                        if (denses[k].plainData[i].IsZero)
                            continue;
                        else
                        {
                            epenv.evaluator.MultiplyPlain(sparse.encData[k], denses[k].plainData[i], step2tmp[taskIndex], epenv.memoryPool);
                            OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                        }
                    }

                    if (tmpSum[taskIndex][i] == null)
                        tmpSum[taskIndex][i] = new Ciphertext(step2tmp[taskIndex]); //CopyCiphertext(step2tmp[taskIndex], epenv);
                        else
                    {
                        epenv.evaluator.Add(tmpSum[taskIndex][i], step2tmp[taskIndex], tmpSum[taskIndex][i]);
                        OperationsCount.Add(ref OperationsCount.Addition, 1);
                    }

                }
            });
            var totalSum = new Ciphertext[l];
            for (int i = 0; i < l; i++)
            {
                totalSum[i] = AllocateCiphertext(env);
                var itemsToAdd = tmpSum.Where(s => s != null).Select(s => s[i]).Where(s => s != null).ToList();
                env.evaluator.AddMany(itemsToAdd, totalSum[i]);
                OperationsCount.Add(ref OperationsCount.AddManyItemCount, itemsToAdd.Count);
            }
            OperationsCount.Add(ref OperationsCount.AddMany, l);
            var res = new AtomicSealBfvEncryptedVector()
            {
                Format = EVectorFormat.dense,
                plainData = null,
                Scale = denses[0].Scale * sparse.Scale,
                IsSigned = sparse.IsSigned,
                encData = totalSum,
                Dim = denses[0].Dim
            };
            if (bothEncrypted) foreach (var t in step1tmp) if (t != null) t.Dispose();
            foreach (var t in step2tmp) if (t != null) t.Dispose();
            foreach (var s in tmpSum)
                if (s != null)
                    foreach (var t in s) if (t != null) t.Dispose();
            return res;
        }


        internal Task<IVector> SparseMultiplyTask(IVector v, ulong colIndex, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => SparseMultiply(v, colIndex, env));
        }

        internal IVector SparseMultiply(IVector v, ulong colIndex, IComputationEnvironment env)
        {
            if (colIndex >= v.Dim) throw new Exception("index exceeds dimension");
            var ev = v as AtomicSealBfvEncryptedVector;
            if (ev.Format != EVectorFormat.sparse) throw new Exception("expecting sparse format");
            if (ev.encData == null && this.encData == null) throw new Exception("at least one argument is expected to be encrypted");
            if (IsSigned != ev.IsSigned) throw new Exception("can't mix signed and unsigned numbers.");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            var t = new AtomicSealBfvEncryptedVector() { Scale = Scale * ev.Scale, Dim = Dim, plainData = null, IsSigned = IsSigned, Format = EVectorFormat.dense };
            if (this.encData != null && ev.encData != null) // both encrypted
            {
                using (var tmp = AllocateCiphertext(eenv))
                {
                    t.encData = new Ciphertext[encData.Length];
                    for (int i = 0; i < encData.Length; i++)
                    {
                        t.encData[i] = AllocateCiphertext(eenv);
                        eenv.evaluator.Multiply(ev.encData[colIndex], encData[i], tmp, eenv.memoryPool);
                        eenv.evaluator.Relinearize(tmp, eenv.relinKeys, t.encData[i], eenv.memoryPool);
                    }
                    OperationsCount.Add(ref OperationsCount.Multiplication, encData.Length);
                    OperationsCount.Add(ref OperationsCount.Relinarization, encData.Length);
                    return t;
                }
            }
            // one is encrypted and the other is not
            if (this.encData == null)
            {
                var c = ev.encData;
                var p = this.plainData;
                var length = p.Length;
                t.encData = new Ciphertext[length];
                for (int i = 0; i < length; i++)
                {
                    t.encData[i] = AllocateCiphertext(eenv);
                    if (p[i].IsZero)
                    {
                        eenv.encryptor.Encrypt(eenv.PlainZero, t.encData[i], eenv.memoryPool);
                        OperationsCount.Add(ref OperationsCount.Encryption, 1);
                    }
                    else
                    {
                        eenv.evaluator.MultiplyPlain(c[colIndex], p[i], t.encData[i], eenv.memoryPool);
                        OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                    }
                }
            }
            else
            {
                var c = this.encData;
                var p = ev.plainData;
                var length = c.Length;
                t.encData = new Ciphertext[length];
                for (int i = 0; i < length; i++)
                {
                    t.encData[i] = AllocateCiphertext(eenv);
                    if (p[colIndex].IsZero)
                    {
                        eenv.encryptor.Encrypt(eenv.PlainZero, t.encData[i], eenv.memoryPool);
                        OperationsCount.Add(ref OperationsCount.Encryption, 1);
                    }
                    else
                    {
                        eenv.evaluator.MultiplyPlain(c[i], p[colIndex], t.encData[i], eenv.memoryPool);
                        OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                    }
                }
            }
            return t;
        }

        static Ciphertext[] Inteleave(Ciphertext[] vecs, int shift, int outputBlockCount, AtomicSealBfvEncryptedEnvironment env)
        {
            int blockSize = (int)env.builder.SlotCount;
            int absShift = (shift < 0) ? -shift : shift;
            bool negativeShift = (shift < 0);
            if (negativeShift && outputBlockCount > 1) throw new Exception("Negative shifts with multiple output blocks are not implemented yet");
            if ((absShift > blockSize / 2) && outputBlockCount > 1) throw new Exception("Shifts of more than half block size with multiple output blocks are not implemented yet");
            if (absShift * vecs.Length > blockSize * outputBlockCount) throw new Exception("not enough room for interleaving");
            var rotatedLower = Enumerable.Range(0, outputBlockCount).Select(x => new ConcurrentBag<Ciphertext>()).ToArray();
            var rotatedUpper = Enumerable.Range(0, outputBlockCount).Select(x => new ConcurrentBag<Ciphertext>()).ToArray();
            Utils.ParallelProcessInEnv(vecs.Length, env, (penv, task, k) =>
                {
                    var epenv = penv as AtomicSealBfvEncryptedEnvironment;
                    var thisShift = shift * k;
                    if (thisShift < 0) thisShift = (blockSize / 2) + thisShift;
                    var inBlockShift = thisShift % blockSize;
                    var startBlock = thisShift / blockSize;
                    var endBlock = (thisShift + absShift) / blockSize;
                    var v = CopyCiphertext(vecs[k], epenv);
                    switch (inBlockShift)
                    {
                        case int s when (s == 0):
                            rotatedLower[startBlock].Add(v);
                            break;
                        case int s when ((s + absShift) < (blockSize / 2)):
                            epenv.evaluator.RotateRowsInplace(v, -thisShift, epenv.galoisKeys, epenv.memoryPool);
                            rotatedLower[startBlock].Add(v);
                            break;
                        case int s when (s >= (blockSize / 2)):
                            if (startBlock == endBlock) // everything is in the same block
                            {
                                epenv.evaluator.RotateRowsInplace(v, -(inBlockShift - (blockSize / 2)), epenv.galoisKeys);
                                rotatedUpper[startBlock].Add(v);
                            }
                            else
                            { // this is the case when the vector is on the boundery between upper and lower and we have to break it in two
                              // 1. shift the message
                                epenv.evaluator.RotateRowsInplace(v, -(inBlockShift - (blockSize / 2)), epenv.galoisKeys);
                                // 2. select the parts
                                int upperPartSize = (inBlockShift + absShift) - (blockSize);
                                var ones = Enumerable.Range(0, upperPartSize).Select(x => 1L).ToList();
                                var v2 = CopyCiphertext(v, epenv);
                                using (var p = new Plaintext(epenv.memoryPool))
                                {
                                    epenv.builder.Encode(ones, p);
                                    epenv.evaluator.MultiplyPlainInplace(v, p);
                                    epenv.evaluator.SubInplace(v2, v);
                                }
                                // 3. add them to the lists
                                rotatedUpper[startBlock].Add(v2);
                                rotatedLower[endBlock].Add(v);

                                // 4. count operations
                                OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                                OperationsCount.Add(ref OperationsCount.Subtraction, 1);
                            }
                            break;
                        default: // this is the case when the vector is on the boundery between lower and upper and we have to break it in two
                            {
                                // 1. shift the message
                                epenv.evaluator.RotateRowsInplace(v, -inBlockShift, epenv.galoisKeys);
                                // 2. select the parts
                                int upperPartSize = (inBlockShift + absShift) - (blockSize / 2);
                                if (upperPartSize > 0)
                                {
                                    var ones = Enumerable.Range(0, upperPartSize).Select(x => 1L).ToList();
                                    var v2 = CopyCiphertext(v, epenv);
                                    using (var p = new Plaintext(epenv.memoryPool))
                                    {
                                        epenv.builder.Encode(ones, p);
                                        epenv.evaluator.MultiplyPlainInplace(v, p, epenv.memoryPool);
                                        epenv.evaluator.SubInplace(v2, v);
                                    }
                                    // 3. add them to the lists
                                    rotatedUpper[startBlock].Add(v);
                                    rotatedLower[startBlock].Add(v2);
                                    // 4. count operations
                                    OperationsCount.Add(ref OperationsCount.PlainMultiplication, 1);
                                    OperationsCount.Add(ref OperationsCount.Subtraction, 1);
                                    OperationsCount.Add(ref OperationsCount.Rotation, 1);
                                    OperationsCount.Add(ref OperationsCount.Addition, 2);
                                } 
                                else
                                {
                                    rotatedLower[startBlock].Add(v);
                                    OperationsCount.Add(ref OperationsCount.Rotation, 1);
                                    OperationsCount.Add(ref OperationsCount.Addition, 2);
                                }
                            }
                            break;
                    }

                });
            var res = new Ciphertext[outputBlockCount];
            Utils.ParallelProcessInEnv(outputBlockCount, env, (penv, taskid, i) =>
            {
                var epenv = penv as AtomicSealBfvEncryptedEnvironment;
                res[i] = AllocateCiphertext(epenv);
                epenv.evaluator.AddMany(rotatedLower[i], res[i]);
                OperationsCount.Add(ref OperationsCount.AddMany, 1);
                OperationsCount.Add(ref OperationsCount.AddManyItemCount, rotatedLower[i].Count);
                OperationsCount.Add(ref OperationsCount.Rotation, rotatedLower[i].Count);
                foreach (var r in rotatedLower[i]) r.Dispose();

                if (rotatedUpper[i].Any())
                {
                using (var t = AllocateCiphertext(epenv))
                    {
                        epenv.evaluator.AddMany(rotatedUpper[i], t);
                        epenv.evaluator.RotateColumnsInplace(t, env.galoisKeys, epenv.memoryPool);
                        epenv.evaluator.AddInplace(res[i], t);
                        OperationsCount.Add(ref OperationsCount.AddMany, 1);
                        OperationsCount.Add(ref OperationsCount.AddManyItemCount, rotatedUpper[i].Count);
                        OperationsCount.Add(ref OperationsCount.Rotation, rotatedUpper[i].Count + 1);
                        foreach (var r in rotatedUpper[i]) r.Dispose();

                    }
                }


            });
            return res;
        }

        public static Task<AtomicSealBfvEncryptedVector> InterleaveTask(AtomicSealBfvEncryptedVector[] vecs, int shift, AtomicSealBfvEncryptedEnvironment env)
        {
            return Task<AtomicSealBfvEncryptedVector>.Factory.StartNew(() => AtomicSealBfvEncryptedVector.Interleave(vecs, shift, env));
        }

        static public AtomicSealBfvEncryptedVector Interleave(AtomicSealBfvEncryptedVector[] vecs, int shift, AtomicSealBfvEncryptedEnvironment env)
        {
            if (vecs[0].Format != EVectorFormat.dense) throw new Exception("Expecting dense vector");
            var blockSize = env.builder.SlotCount;
//            if (vecs[0].Dim > blockSize / 2)  throw new Exception("no support for inteleave of data that is greater than half block");
            int outputBlocks = 1;
            if (shift > 0)
            {
                var totalLength = vecs[0].Dim * (ulong)vecs.Length;
                outputBlocks = (int)Math.Ceiling(totalLength / (double)blockSize);

            }
            return new AtomicSealBfvEncryptedVector()
            {
                encData = Inteleave(vecs.Select(v => v.encData[0]).ToArray(), shift, outputBlocks, env),
                plainData = null,
                Dim = vecs[0].Dim,
                Scale = vecs[0].Scale,
                IsSigned = vecs[0].IsSigned,
                Format = EVectorFormat.dense
            };
        }

        static public Task<AtomicSealBfvEncryptedVector> StackTask(AtomicSealBfvEncryptedVector[] vecs, AtomicSealBfvEncryptedEnvironment env)
        {
            return Task<AtomicSealBfvEncryptedVector>.Factory.StartNew(() => Stack(vecs, env));
        }
        static public AtomicSealBfvEncryptedVector Stack(AtomicSealBfvEncryptedVector[] vecs, AtomicSealBfvEncryptedEnvironment env)
        {
            var res = Interleave(vecs, (int)vecs[0].Dim, env);
            res.Dim = vecs[0].Dim * (ulong)vecs.Length;
            return res;
        }

        public Task<IVector> PointwiseMultiplyTask(IVector v, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => PointwiseMultiply(v, env));
        }

        /// <summary>
        /// This is a special multiplication setting when one of the vectors is sparse of dimension 1 which means that we just wanted to multiply the other one by a constant factor
        /// </summary>
        /// <param name="v"> a sparse vector of dimension 1</param>
        /// <param name="env"></param>
        /// <returns></returns>
        IVector PointwiseMultiplySparseDimOne(AtomicSealBfvEncryptedVector ev, AtomicSealBfvEncryptedEnvironment eenv)
        {
            var t = new AtomicSealBfvEncryptedVector() { Scale = Scale * ev.Scale, Dim = Dim, Format = Format, IsSigned = IsSigned, plainData = null };
            if (this.encData != null && ev.encData != null) // both encrypted
            {
                t.encData = new Ciphertext[encData.Length];
                Utils.ParallelProcessInEnv(encData.Length, eenv, (localEnv, task, i) =>
                {
                    var lenv = localEnv as AtomicSealBfvEncryptedEnvironment;
                    using (var tmp = AllocateCiphertext(lenv))
                    {
                        t.encData[i] = AllocateCiphertext(lenv);
                        lenv.evaluator.Multiply(ev.encData[0], encData[i], tmp, lenv.memoryPool);
                        lenv.evaluator.Relinearize(tmp, lenv.relinKeys, t.encData[i], lenv.memoryPool);
                    }
                });
                OperationsCount.Add(ref OperationsCount.Multiplication, t.encData.Length);
                OperationsCount.Add(ref OperationsCount.Relinarization, t.encData.Length);
                return t;
            }
            // one is encrypted and the other is not
            var enc = encData ?? ev.encData;
            var plain = plainData ?? ev.plainData;
            var flag = (encData != null);
            t.encData = new Ciphertext[enc.Length];
            Utils.ParallelProcessInEnv(t.encData.Length, eenv, (localEnv, task, i) =>
            {
                var lenv = localEnv as AtomicSealBfvEncryptedEnvironment;
                t.encData[i] = AllocateCiphertext(lenv);
                lenv.evaluator.MultiplyPlain(enc[flag ? i : 0], plain[flag ? 0 : i], t.encData[i], lenv.memoryPool);
            });
            OperationsCount.Add(ref OperationsCount.PlainMultiplication, t.encData.Length);

            return t;


        }


        public IVector PointwiseMultiply(IVector v, IComputationEnvironment env) 
        {
            var ev = v as AtomicSealBfvEncryptedVector;
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            if (IsSigned != ev.IsSigned) throw new Exception("Can't mix signed and unsigned numbers.");
            if (this.plainData != null && ev.plainData != null)
            {
                throw new Exception("multiplying two plaintexts is not implemented");
            }
            if (Dim == 1 && Format == EVectorFormat.sparse)
                return ev.PointwiseMultiplySparseDimOne(this, eenv);
            if (ev.Dim == 1 && ev.Format == EVectorFormat.sparse)
                return PointwiseMultiplySparseDimOne(ev, eenv);
            // we expect both vectors to be of same dimension and same format
            if (Dim != v.Dim) throw new Exception("Dimensions do not match");
            if (Format != ev.Format) throw new Exception("Format mismatch");
            var t = new AtomicSealBfvEncryptedVector() { Scale = Scale * ev.Scale, Dim = Dim, Format = Format, IsSigned = IsSigned, plainData = null };
            if (this.encData != null && ev.encData != null) // both encrypted
            {
                t.encData = new Ciphertext[ev.encData.Length];
                Utils.ParallelProcessInEnv(t.encData.Length, eenv, (localEnv, task, i) =>
                {
                    var lenv = localEnv as AtomicSealBfvEncryptedEnvironment;
                    using (var tmp = AllocateCiphertext(lenv))
                    {
                        t.encData[i] = AllocateCiphertext(lenv);
                        lenv.evaluator.Multiply(ev.encData[i], encData[i], tmp, lenv.memoryPool);
                        lenv.evaluator.Relinearize(tmp, lenv.relinKeys, t.encData[i], lenv.memoryPool);
                    }
                });
                OperationsCount.Add(ref OperationsCount.Multiplication,t.encData.Length);
                OperationsCount.Add(ref OperationsCount.Relinarization, t.encData.Length);
                return t;
            }
            // one is encrypted and the other is not
            var enc = encData ?? ev.encData;
            var plain = plainData ?? ev.plainData;
            t.encData = new Ciphertext[enc.Length];
            Utils.ParallelProcessInEnv(t.encData.Length, eenv, (localEnv, task, i) =>
            {
                var lenv = localEnv as AtomicSealBfvEncryptedEnvironment;
                t.encData[i] = AllocateCiphertext(lenv);
                lenv.evaluator.MultiplyPlain(enc[i], plain[i], t.encData[i], lenv.memoryPool);
            });
            OperationsCount.Add(ref OperationsCount.PlainMultiplication, t.encData.Length);

            return t;
        }

        void RotateRowsAndAdd(Ciphertext c, int steps, AtomicSealBfvEncryptedEnvironment eenv, Ciphertext agg, Ciphertext tmp)
        {
            eenv.evaluator.RotateRows(c, -steps, eenv.galoisKeys, tmp, eenv.memoryPool);
            eenv.evaluator.AddInplace(agg, tmp);
            OperationsCount.Add(ref OperationsCount.Rotation, 1);
            OperationsCount.Add(ref OperationsCount.Addition, 1);
        }


        public Task<IVector> SumAllSlotsTask(IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => SumAllSlots(Int32.MaxValue, env));
        }


        public IVector SumAllSlots(IComputationEnvironment env) => SumAllSlots(env, null);
        public IVector SumAllSlots(IComputationEnvironment env, int? ForceOutputInColumn = null)
        {
            return SumAllSlots(Int32.MaxValue, env, ForceOutputInColumn);
        }

        public Task<IVector> SumAllSlotsTask(ulong length, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => SumAllSlots(length, env));
        }
        public IVector SumAllSlots(ulong length, IComputationEnvironment env) => SumAllSlots(length, env, null);
        public IVector SumAllSlots(ulong length, IComputationEnvironment env, int? ForceOutputInColumn = null)
        {
            if (Format != EVectorFormat.dense) throw new Exception("Expecting dense vector format");
            if (length != Int32.MaxValue && ForceOutputInColumn != null) throw new Exception("forcing output in a column works only when doing complete sum");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            if (plainData != null) throw new Exception("SumAllSlots can be applied to encrypted data only");
            if (length <= 0) throw new Exception("Can't sum over less then one element");
            if (length == 1) return this;
            bool discardSumlong = true;
            Ciphertext sumLong = null;

            if (encData.Length > 1)
            {
                sumLong = AllocateCiphertext(eenv);
                eenv.evaluator.AddMany(encData, sumLong);
                OperationsCount.Add(ref OperationsCount.AddMany, 1);
                OperationsCount.Add(ref OperationsCount.AddManyItemCount, encData.Length);
            } else
            {
                sumLong = CopyCiphertext(encData[0], eenv);
            }
            Ciphertext sum = null;
            if (length >= eenv.builder.SlotCount / 2)
            {
                using (var rotatedCol = AllocateCiphertext(eenv))
                {
                    eenv.evaluator.RotateColumns(sumLong, eenv.galoisKeys, rotatedCol, eenv.memoryPool);
                    OperationsCount.Add(ref OperationsCount.Rotation, 1);
                    sum = AllocateCiphertext(eenv);
                    eenv.evaluator.Add(sumLong, rotatedCol, sum);
                    OperationsCount.Add(ref OperationsCount.Addition, 1);
                    length = eenv.builder.SlotCount / 2;
                    discardSumlong = true;
                }
            }
            else
            {
                sum = sumLong;
                discardSumlong = false;
            }
            using (var tmp = AllocateCiphertext(eenv))
            {
                for (ulong steps = 1; steps < length; steps *= 2)
                {
                    RotateRowsAndAdd(sum, (int)steps, eenv, sum, tmp);
                }
            }
            if (discardSumlong) sumLong.Dispose();
            if (ForceOutputInColumn != null)
            {
                var col = ForceOutputInColumn.Value;
                using (var p = new Plaintext(eenv.memoryPool))
                {
                    eenv.builder.Encode(Enumerable.Range(0, col + 1).Select(x => (x == col) ? 1L : 0L).ToList(), p);
                    eenv.evaluator.MultiplyPlainInplace(sum, p, eenv.memoryPool);
                }
                length = 1;
            }
            return new AtomicSealBfvEncryptedVector()
            {
                IsSigned = IsSigned,
                Scale = Scale,
                Dim = (length >= eenv.builder.SlotCount / 2) ? 1 : this.Dim,
                encData = new Ciphertext[] { sum },
                plainData = null,
                Format = (length >= eenv.builder.SlotCount) ? EVectorFormat.sparse : EVectorFormat.dense
            };
        }


        public Task<IVector> DotProductTask(IVector v, IComputationEnvironment env, int? ForceOutputInColumn = null)
        {
            return Task<IVector>.Factory.StartNew(() => DotProduct(v, env, ForceOutputInColumn));
        }

        public IVector DotProduct(IVector v, IComputationEnvironment env) => DotProduct(v, env, null);
        public IVector DotProduct(IVector v, IComputationEnvironment env, int? ForceOutputInColumn = null)
        {
            using (var mul = (AtomicSealBfvEncryptedVector)PointwiseMultiply(v, env))
                return mul.SumAllSlots(env, ForceOutputInColumn);
        }
        public Task<IVector> DotProductTask(IVector v, ulong length, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => DotProduct(v, length, env));
        }
        public IVector DotProduct(IVector v, ulong length, IComputationEnvironment env)
        {
            using (var mul = (AtomicSealBfvEncryptedVector)PointwiseMultiply(v, env))
                return mul.SumAllSlots(length, env);
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
            if (Dim != v.Dim) throw new Exception("Dimensions do not match");
            var ev = v as AtomicSealBfvEncryptedVector;
            if (Format != ev.Format) throw new Exception("Format mismatch");
            if (IsSigned != ev.IsSigned) throw new Exception("can't mix signed and unsigned numbers.");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            if (this.plainData != null && ev.plainData != null)
            {
                throw new Exception("adding two plaintexts is not supported");
            }
            var t = new AtomicSealBfvEncryptedVector() { Scale = Scale, Dim = Dim, Format = Format, IsSigned = IsSigned };
            if (this.encData != null && ev.encData != null) // both encrypted
            {
                t.encData = new Ciphertext[ev.encData.Length];
                Utils.ParallelProcessInEnv(encData.Length, eenv, (penv, task, i) =>
                {
                    var epenv = penv as AtomicSealBfvEncryptedEnvironment;
                    t.encData[i] = AllocateCiphertext(epenv);
                    epenv.evaluator.Add(ev.encData[i], encData[i], t.encData[i]);
                });
                OperationsCount.Add(ref OperationsCount.Addition, encData.Length);
                return t;
            }
            // one is encrypted and the other is not
            var enc = encData ?? ev.encData;
            var plain = plainData ?? ev.plainData;
            t.encData = new Ciphertext[enc.Length];
            Utils.ParallelProcessInEnv(plain.Length, eenv, (penv, task, i) =>
            {
                var epenv = penv as AtomicSealBfvEncryptedEnvironment;

                t.encData[i] = AllocateCiphertext(epenv);
                epenv.evaluator.AddPlain(enc[i], plain[i], t.encData[i]);
            });
            OperationsCount.Add(ref OperationsCount.PlainAddition, plain.Length);
            return t;

        }

        public Task<Vector<double>> DecryptTask(IComputationEnvironment env)
        {
            return Task<Vector<double>>.Factory.StartNew(() => Decrypt(env));
        }
        public Vector<double> Decrypt(IComputationEnvironment env)
        {
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            var plain = new Plaintext(eenv.memoryPool);
            List<double> res = new List<double>();
            var mod = (double)eenv.parameters.PlainModulus.Value;
            var length = (encData == null) ? plainData.Length : encData.Length;
            for (int i = 0; i < length; i++)
            {
                if (encData != null)
                {
                    CryptoTracker.TestBudget(encData[i], eenv.decryptor);
                    eenv.decryptor.Decrypt(encData[i], plain);
                    OperationsCount.Add(ref OperationsCount.Decryption, 1);
                }
                else
                    plain = plainData[i];
                if (Format == EVectorFormat.dense)
                {
                    List<ulong> local = new List<ulong>();
                    eenv.builder.Decode(plain, local);
                    int left = (int)Dim - res.Count;
                    var t = local.Take((left > local.Count) ? local.Count : left)
                        .Select(v => (IsSigned && v * 2 > eenv.parameters.PlainModulus.Value) ? v - mod : v)
                        .Select(v => v / Scale);
                    res.AddRange(t);
                }
                else
                {
                    // sparse format
                    var w = new BigUInt(plain.ToString());
                    var bi = (BigInteger)w.ToBigInteger();
                    var v = (double)bi;
                    res.Add((IsSigned && 2 * v > eenv.parameters.PlainModulus.Value) ? (v - mod) / Scale : v / Scale);
                }
            }
            return Vector<double>.Build.DenseOfEnumerable(res);
        }

        public Task<IEnumerable<BigInteger>> DecryptFullPrecisionTask(IComputationEnvironment env)
        {
            return Task<IEnumerable<BigInteger>>.Factory.StartNew(() => DecryptFullPrecision(env));
        }
        public IEnumerable<BigInteger> DecryptFullPrecision(IComputationEnvironment env)
        {
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            var plain = new Plaintext(eenv.memoryPool);
            var res = new List<BigInteger>();
            var mod = eenv.parameters.PlainModulus.Value;
            var length = (encData == null) ? plainData.Length : encData.Length;
            for (int i = 0; i < length; i++)
            {
                if (encData != null)
                {
                    CryptoTracker.TestBudget(encData[i], eenv.decryptor);
                    eenv.decryptor.Decrypt(encData[i], plain);
                    OperationsCount.Add(ref OperationsCount.Decryption, 1);
                }
                else
                    plain = plainData[i];
                if (Format == EVectorFormat.dense)
                {
                    List<ulong> local = new List<ulong>();
                    eenv.builder.Decode(plain, local);
                    int left = (int)Dim - res.Count;
                    var t = local.Take((left > local.Count) ? local.Count : left)
                        .Select(v => (IsSigned && v * 2 > eenv.parameters.PlainModulus.Value) ? v - mod : v)
                        .Select(v => new BigInteger(v));
                    res.AddRange(t);
                }
                else
                {
                    // sparse format
                    var w = new BigUInt(plain.ToString());
                    var bi = (BigInteger)w.ToBigInteger();
                    var v = (double)bi;
                    res.Add(new BigInteger((IsSigned && 2 * v > eenv.parameters.PlainModulus.Value) ? (v - mod)  : v ));
                }
            }
            return res;
        }



        Plaintext[] VectorToPlaintext(Vector<double> v, AtomicSealBfvEncryptedEnvironment eenv)
        {
            List<Plaintext> lst = new List<Plaintext>();
            int start = 0;
            int length = v.Count;
            int slots = (int)eenv.builder.SlotCount;
            if (Scale == 0) Scale = 1;
            var values = v.Multiply(Scale).PointwiseRound()
                .Select(x => (ulong)((!IsSigned || x >= 0) ? x : eenv.parameters.PlainModulus.Value + x)).ToList();
            while (start < length)
            {
                if (Format == EVectorFormat.dense)
                {
                    var size = (start + slots <= length) ? slots : length - start;
                    var chunk = values.GetRange(start, size);
                    var p = new Plaintext(eenv.memoryPool);
                    eenv.builder.Encode(chunk, p);
                    lst.Add(p);
                    start += size;
                }
                else
                { // Sparse Format
                    var p = new Plaintext(values[start].ToString("X"), eenv.memoryPool);
                    lst.Add(p);
                    start++;
                }
            }
            return lst.ToArray();
        }

        Plaintext[] VectorToPlaintext(UInt64[] v, AtomicSealBfvEncryptedEnvironment eenv)
        {
            List<Plaintext> lst = new List<Plaintext>();
            int start = 0;
            int length = v.Length;
            int slots = (int)eenv.builder.SlotCount;
            var values = v.ToList();
            while (start < length)
            {
                if (Format == EVectorFormat.dense)
                {
                    int size = (start + slots <= length) ? slots : length - start;
                    var chunk = values.GetRange(start, size);
                    var p = new Plaintext(eenv.memoryPool);
                    eenv.builder.Encode(chunk, p);
                    lst.Add(p);
                    start += size;
                }
                else
                { // Sparse Format
                    var p = new Plaintext(values[start].ToString("X"), eenv.memoryPool);
                    lst.Add(p);
                    start++;
                }
            }
            return lst.ToArray();
        }


        Plaintext DoubleToPlaintext(double v, AtomicSealBfvEncryptedEnvironment eenv)
        {

            if (Scale == 0) Scale = 1;
            var value = Math.Round(v * Scale);
            var unsigned = (ulong)((!IsSigned || value >= 0) ? value : eenv.parameters.PlainModulus.Value + value);
            var plain = new Plaintext(unsigned.ToString("X"), eenv.memoryPool);
            return plain;
        }

        void Plain(Vector<double> v, EVectorFormat Format, IComputationEnvironment env)
        {
            OperationsCount.Add(ref OperationsCount.Plain, 1);
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            this.Format = Format;
            plainData = VectorToPlaintext(v, eenv);
            encData = null;
            Dim = (ulong)v.Count;
        }
        void Plain(UInt64[] v, EVectorFormat Format, IComputationEnvironment env)
        {
            OperationsCount.Add(ref OperationsCount.Plain, 1);
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            this.Format = Format;
            plainData = VectorToPlaintext(v, eenv);
            encData = null;
            Dim = (ulong) v.Length;
        }

        void Encrypt(Vector<double> v, EVectorFormat format, IComputationEnvironment env)
        {
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            this.Format = format;
            var plain = VectorToPlaintext(v, eenv);
            encData = new Ciphertext[plain.Length];
            for (int i = 0; i < plain.Length; i++)
            {
                encData[i] = AllocateCiphertext(eenv);
                eenv.encryptor.Encrypt(plain[i], encData[i], eenv.memoryPool);
            }
            plainData = null;
            Dim = (ulong)v.Count;
            OperationsCount.Add(ref OperationsCount.Encryption, plain.Length);
        }

        void Encrypt(UInt64[] v, EVectorFormat format, IComputationEnvironment env)
        {
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            this.Format = format;
            var plain = VectorToPlaintext(v, eenv);
            encData = new Ciphertext[plain.Length];
            for (int i = 0; i < plain.Length; i++)
            {
                encData[i] = AllocateCiphertext(eenv);
                eenv.encryptor.Encrypt(plain[i], encData[i], eenv.memoryPool);
            }
            plainData = null;
            Dim = (ulong)v.Length;
            OperationsCount.Add(ref OperationsCount.Encryption, plain.Length);
        }

        public Task<IVector> SubtractTask(IVector v, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Subtract(v, env));
        }
        public IVector Subtract(IVector v, IComputationEnvironment env)
        {
            if (v.Scale == 0) return this;
            if (Scale != v.Scale) throw new Exception("Scales do not match.");
            if (Dim != v.Dim) throw new Exception("Dimensions do not match");
            var ev = v as AtomicSealBfvEncryptedVector;
            if (Format != ev.Format) throw new Exception("Format mismatch");
            if (IsSigned != ev.IsSigned) throw new Exception("Can't mix signed and unsigned numbers.");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            if (this.plainData != null)
            {
                throw new Exception("the first argument for subtraction must be encrypted");
            }
            var t = new AtomicSealBfvEncryptedVector() { Scale = Scale, Dim = Dim, Format = Format, IsSigned = IsSigned };
            t.encData = new Ciphertext[encData.Length];
            if (ev.encData != null) // both encrypted
            {
                for (int i = 0; i < encData.Length; i++)
                {
                    t.encData[i] = AllocateCiphertext(eenv);
                    eenv.evaluator.Sub(encData[i], ev.encData[i], t.encData[i]);
                }
                OperationsCount.Add(ref OperationsCount.Subtraction, encData.Length);
                return t;
            }
            // this is encrypted but ev is not
            for (int i = 0; i < encData.Length; i++)
            {
                t.encData[i] = AllocateCiphertext(eenv);
                eenv.evaluator.SubPlain(encData[i], ev.plainData[i], t.encData[i]);
            }
            OperationsCount.Add(ref OperationsCount.PlainSubtraction, encData.Length);
            return t;
        }

        public void Write(StreamWriter str)
        {
            str.WriteLine("<Start EncryptedVector>");
            str.WriteLine(Scale);
            str.WriteLine(IsSigned);
            str.WriteLine(Enum.GetName(Format.GetType(), Format));
            str.WriteLine(Dim);
            using (MemoryStream mem = new MemoryStream())
            {
                if (plainData == null)
                {
                    str.WriteLine("Encrypted");
                    str.WriteLine(encData.Length);
                    for (int i = 0; i < encData.Length; i++) encData[i].Save(mem);
                }
                else
                {
                    str.WriteLine("Plain");
                    str.WriteLine(plainData.Length);
                    for (int i = 0; i < plainData.Length; i++) plainData[i].Save(mem);
                }
                mem.Flush();
                mem.Position = 0;
                str.WriteLine(Convert.ToBase64String(mem.ToArray(), Base64FormattingOptions.None));

                str.WriteLine("<End EncryptedVector>");
                str.Flush();
            }
        }

        public static AtomicSealBfvEncryptedVector Read(StreamReader str, AtomicSealBfvEncryptedEnvironment env)
        {
            var vct = new AtomicSealBfvEncryptedVector();
            var line = str.ReadLine();
            if (line != "<Start EncryptedVector>")
                throw new Exception("Bad stream format.");
            vct.Scale = Double.Parse(str.ReadLine());
            vct.IsSigned = Boolean.Parse(str.ReadLine());
            vct.Format = (EVectorFormat)Enum.Parse(vct.Format.GetType(), str.ReadLine());
            vct.Dim = ulong.Parse(str.ReadLine());
            var mode = str.ReadLine();
            var length = int.Parse(str.ReadLine());
            var base64 = str.ReadLine();
            using (var mem = new MemoryStream(Convert.FromBase64String(base64)))
            {
                switch (mode)
                {
                    case "Encrypted":
                        vct.plainData = null;
                        vct.encData = new Ciphertext[length];
                        for (int i = 0; i < length; i++)
                        {
                            vct.encData[i] = AllocateCiphertext(env);
                            vct.encData[i].Load(env.context, mem);
                        }
                        break;
                    case "Plain":
                        vct.encData = null;
                        vct.plainData = new Plaintext[length];
                        for (int i = 0; i < length; i++)
                        {
                            vct.plainData[i] = new Plaintext(env.memoryPool);
                            vct.plainData[i].Load(env.context, mem);
                        }
                        break;
                    default: throw new Exception("unknown format");
                }
            }
            line = str.ReadLine();
            if (line != "<End EncryptedVector>")
                throw new Exception("Bad stream format.");
            return vct;
        }

        public static AtomicSealBfvEncryptedVector GenerateSparseOfArray(AtomicSealBfvEncryptedVector[] encryptedVector, IComputationEnvironment env)
        {
            var res = new AtomicSealBfvEncryptedVector()
            {
                Scale = encryptedVector[0].Scale,
                Dim = (ulong)encryptedVector.Length,
                Format = EVectorFormat.sparse,
                IsSigned = encryptedVector[0].IsSigned,
                plainData = null,
                encData = encryptedVector.Select(e => CopyCiphertext(e.encData[0], (AtomicSealBfvEncryptedEnvironment) env)).ToArray()
            };
            return res;
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
            ulong shift = 1;
            while (shift < Dim) shift *= 2;
            if (encData == null) throw new Exception("Duplicate operates only on encrypted data");
            if (Format == EVectorFormat.sparse) throw new Exception("Duplicate operates only on dense vectors");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            if (shift * count > eenv.builder.SlotCount) throw new Exception("Packed vector must fit in a single ciphertext");
            var res = CopyCiphertext(encData[0], eenv);
            bool columnRotated = false;
            using (var rotator = CopyCiphertext(encData[0], eenv))
            using (var tmp = AllocateCiphertext(eenv))
            {
                for (ulong i = 1; i < count; i++)
                {
                    int targetShiftSize = (int) ( i * shift);
                    if (targetShiftSize * 2 >= (int)eenv.builder.SlotCount)
                    {
                        if (!columnRotated)
                        {
                            columnRotated = true;
                            eenv.evaluator.RotateColumns(encData[0], eenv.galoisKeys, rotator, eenv.memoryPool);
                            OperationsCount.Add(ref OperationsCount.Rotation, 1);
                        }
                        targetShiftSize -= (int)eenv.builder.SlotCount / 2;
                    }
                    RotateRowsAndAdd(rotator, targetShiftSize, eenv, res, tmp);
                }
            }
            return new AtomicSealBfvEncryptedVector()
            {
                IsSigned = IsSigned,
                Scale = Scale,
                Dim = count * shift,
                encData = new Ciphertext[] { res },
                plainData = null,
                Format = EVectorFormat.dense
            };
        }

        public Task<IVector> RotateTask(int amount, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Rotate(amount, env));
        }
        public IVector Rotate(int amount, IComputationEnvironment env)
        {
            if (encData == null) throw new Exception("Rotate operates only on encrypted data");
            if (Format == EVectorFormat.sparse) throw new Exception("Rotate operates only on dense vectors");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            var res = CopyCiphertext(encData[0], eenv);
            eenv.evaluator.RotateRowsInplace(res, amount, eenv.galoisKeys, eenv.memoryPool);
            return new AtomicSealBfvEncryptedVector()
            {
                IsSigned = IsSigned,
                Scale = Scale,
                Dim = Dim,
                encData = new Ciphertext[] { res },
                plainData = null,
                Format = EVectorFormat.dense
            };
        }
        internal Task<IVector> PermuteTask(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env)
        {
            return Task<IVector>.Factory.StartNew(() => Permute(selections, shifts, outputDim, env));
        }

        public IVector Permute(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env)
        {
            if (Format != EVectorFormat.dense) throw new Exception("Permute works only on dense vectors");
            if (selections.Length != shifts.Length) throw new Exception("number of selection vectors and number of shifts does not match");
            if (plainData != null) throw new Exception("can permute only encrypted vectors");
            if (encData.Length > 1) throw new Exception("can permute only a single block");
            var eenv = env as AtomicSealBfvEncryptedEnvironment;
            Ciphertext res = null;
            int firstAvaialbleSelection = -1;
            using (var t = AllocateCiphertext(eenv))
            {
                for (int i = 0; i < selections.Length; i++)
                {
                    if (selections[i] == null) continue;
                    if (firstAvaialbleSelection < 0) firstAvaialbleSelection = i;
                    if (selections[i].Dim != Dim) throw new Exception("dimension of selection vector does not match dimension of data vector");
                    if (selections[i].Scale != selections[firstAvaialbleSelection].Scale) throw new Exception("scales of all selection vectors should be the same");
                    var s = selections[i] as AtomicSealBfvEncryptedVector;
                    if (s.plainData != null)
                        eenv.evaluator.MultiplyPlain(encData[0], s.plainData[0], t, eenv.memoryPool);
                    else
                        eenv.evaluator.Multiply(encData[0], s.encData[0], t, eenv.memoryPool);
                    eenv.evaluator.RotateRowsInplace(t, shifts[i], eenv.galoisKeys, eenv.memoryPool);
                    if (res == null)
                        res = CopyCiphertext(t, eenv);
                    else
                        eenv.evaluator.AddInplace(res, t);
                }
            }
            if (firstAvaialbleSelection < 0) throw new Exception("permuting with no selected values is illigal");
            return new AtomicSealBfvEncryptedVector()
            {
                IsSigned = IsSigned,
                Scale = Scale * selections[firstAvaialbleSelection].Scale,
                Dim = outputDim,
                encData = new Ciphertext[] { res },
                plainData = null,
                Format = EVectorFormat.dense
            };
        }
    }
}
