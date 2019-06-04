// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System.IO;
using MathNet.Numerics.Data.Text;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Linq;

namespace HEWrapper
{
    public class RawVector : IVector
    {
        Vector<double> v;
        public double Scale { get; private set; } = 0;
        public bool IsSigned { get; set; } = true;
        public EVectorFormat Format { get; set; }
        public static double Max = 0;

        public ulong BlockSize { get; private set; }

        public RawVector(Vector<double> v, double scale, ulong BlockSize)
        {
            if (v != null && Double.IsInfinity(v.InfinityNorm())) throw new Exception("infinity");
            this.Scale = scale;

            this.v = (v * Scale).PointwiseRound();
            this.BlockSize = BlockSize;
        }

        public RawVector(IEnumerable<BigInteger> v, ulong BlockSize)
        {
            this.Scale = 1;

            this.v = Vector<double>.Build.DenseOfEnumerable(v.Select(x => Math.Round((double)x * Scale)));
            this.BlockSize = BlockSize;

        }
        public RawVector(double v, double scale, ulong BlockSize)
        {
            if (Double.IsInfinity(v)) throw new Exception("infinity");
            this.Scale = scale;

            this.v = Vector<double>.Build.DenseOfEnumerable(new double[] { Math.Round(v * Scale) });
            this.BlockSize = BlockSize;
        }

        public RawVector(RawVector vr)
        {
            this.v = Vector<double>.Build.DenseOfVector(vr.v);
            this.Scale = vr.Scale;
            this.BlockSize = vr.BlockSize;

        }

        private RawVector(ulong BlockSize)
        {
            this.BlockSize = BlockSize;
        }


        public void Dispose()
        {
            v = null;
        }

        public Vector<double> Decrypt(IComputationEnvironment env)
        {
            Max = Math.Max(Max, v.InfinityNorm() );
            return v / Scale;
        }

        public IEnumerable<BigInteger> DecryptFullPrecision(IComputationEnvironment env)
        {
            Max = Math.Max(Max, v.InfinityNorm());
            return v.Select(x => new BigInteger((IsSigned) ? x : Math.Abs(x)));
        }


        void Create(Vector<double> v, double scale)
        {
            this.Scale = scale;

            this.v = (v * Scale).PointwiseRound();
        }

        static public RawVector Read(StreamReader str)
        {
            RawVector vct = new RawVector(0)
            {
                BlockSize = ulong.Parse(str.ReadLine()),
                Scale = Double.Parse(str.ReadLine()),
                v = DelimitedReader.Read<double>(str).Column(0),
            };
            return vct;
        }

        public void Write(StreamWriter str)
        {
            str.WriteLine(BlockSize);
            str.WriteLine(Scale);
            DelimitedWriter.Write<double>(str, v.ToColumnMatrix());
            str.Flush();
        }


        public IVector Add(IVector v, IComputationEnvironment env)
        {
            if (Scale == 0) return v;
            if (v.Scale == 0) return this;
            if (Scale != v.Scale) throw new Exception("Scales do not match.");
            var vr = v as RawVector;
            var res = this.v.Add(vr.v);
            var t = new RawVector(res, 1, BlockSize);
            t.RegisterScale(Scale);
            return t;
        }
        public IVector Subtract(IVector v, IComputationEnvironment env)
        {
            if (v.Scale == 0) return this;
            var vr = v as RawVector;
            if (Scale != 0 && Scale != vr.Scale) throw new Exception("Scales do not match.");
            var res = this.v.Subtract(vr.v);
            var t = new RawVector(res, 1, BlockSize);
            t.RegisterScale(Scale);
            return t;
        }

        public IVector Multiply(double x, IComputationEnvironment env)
        {
            var res = this.v.Multiply(x);
            var t = new RawVector(res, 1, BlockSize);
            t.RegisterScale(Scale);
            return t;
        }

        public IVector DotProduct(IVector v, IComputationEnvironment env)
        {
            var vr = v as RawVector;
            var dot = this.v.DotProduct(vr.v);
            var vdot = Vector<double>.Build.DenseOfEnumerable(new double[] { dot });
            var res = new RawVector(vdot, 1, BlockSize);
            res.RegisterScale(Scale * vr.Scale );
            return res;

        }

       void Rotate(Vector<double> w, int length, Vector<double> res)
        {
            res.Clear();
            if (w.Count > res.Count - length )
            {
                res.SetSubVector(length, res.Count - length, w);
                res.SetSubVector(0, w.Count - (res.Count - length), w.SubVector(res.Count - length, w.Count - (res.Count - length)));
            } else
            {
                res.SetSubVector(length, w.Count, w);
            }
        }

        public IVector DotProduct(IVector w, ulong length, IComputationEnvironment env)
        {
            var wr = w as RawVector;
            var res = v.PointwiseMultiply(wr.v);
            var shift = Vector<double>.Build.Dense((int)Dim);
            ulong skip = 1;
            while (skip < length)
            {
                Rotate(res, (int)skip, shift);
                res = res.Add(shift);
                skip *= 2;
            }



            var resVector = new RawVector(res, 1, BlockSize);
            resVector.RegisterScale(Scale * wr.Scale);
            return resVector;

        }
        public IVector PointwiseMultiply(IVector v, IComputationEnvironment env)
        {
            var vr = v as RawVector;
            Vector<double> mul = null;
            if (this.v.Count == vr.v.Count)
                mul = this.v.PointwiseMultiply(vr.v);
            else if (this.v.Count == 1 && this.Format == EVectorFormat.sparse) // multiplying by constant
                mul = vr.v.Multiply(this.v[0]);
            else if (vr.v.Count == 1 && vr.Format == EVectorFormat.sparse) // mutliplying by constant
                mul = this.v.Multiply(vr.v[0]);
            else
                throw new Exception("Vectors dimensions do not match");
            var res = new RawVector(mul, 1, BlockSize);
            res.RegisterScale(Scale * vr.Scale );
            return res;
        }

        public IVector SumAllSlots(IComputationEnvironment env)
        {
            var sum = v.Sum();
            var vsum = Vector<double>.Build.DenseOfEnumerable(new double[] { sum });
            var res = new RawVector(vsum, 1, BlockSize);
            res.RegisterScale(Scale);
            return res;

        }

        public void RegisterScale(double scale)
        {
            this.Scale = scale;
        }

        public IVector Duplicate(ulong count, IComputationEnvironment env)
        {
            int shift = 1;
            while (shift < (int)Dim) shift *= 2;
            var w = Vector<double>.Build.Dense(shift * (int)count);
            for (int i = 0; i < (int)count; i++)
            {
                w.SetSubVector(i * shift, (int)Dim, v);
            }
            return new RawVector(w / Scale, Scale, BlockSize);
        }

        Vector<double> Rotate(Vector<double> vec, int amount)
        {
            var w = Vector<double>.Build.Dense(v.Count);
            for (int i = 0; i < vec.Count; i++)
            {
                int k = (i + amount) % (int)BlockSize;
                if (k < 0) k += (int)BlockSize;
                if (k < v.Count) w[i] = vec[k];
            }
            return w;

        }
        public IVector Rotate(int amount, IComputationEnvironment env)
        {
            var res = new RawVector(Rotate(v, amount), 1, BlockSize);
            res.RegisterScale(Scale);
            return res;
        }

        public IVector Permute(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env)
        {
            if (selections.Length != shifts.Length) throw new Exception("number of selection vectors and number of shifts does not match");
            var res = Vector<double>.Build.Dense((int)Dim);
            for (int i = 0; i < selections.Length; i++)
            {
                if (selections[i] == null) continue;
                if (selections[i].Dim != Dim) throw new Exception("dimension of selection vector does not match dimension of data vector");
                if (selections[i].Scale != selections[0].Scale) throw new Exception("scales of all selection vectors should be the same");
                var s = selections[i] as RawVector;
                var t = v.PointwiseMultiply(s.v);
                res = res.Add(Rotate(t, shifts[i]));
            }
            var resVec = new RawVector(res.SubVector(0, (int)outputDim), 1, BlockSize);
            resVec.RegisterScale(Scale * selections[0].Scale);
            return resVec;
        }

        public ulong Dim { get { return (v == null) ? 0 : (ulong)v.Count; } }
        public object Data { get { return Vector<double>.Build.DenseOfVector(v); } }
        public bool IsEncrypted { get { return false; } }

    }
}
