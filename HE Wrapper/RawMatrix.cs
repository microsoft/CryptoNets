// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System;
using System.Linq;
using MathNet.Numerics.Data.Text;

namespace HEWrapper
{
    public class RawMatrix : IMatrix
    {
        public EMatrixFormat Format { get; private set; } 

        Matrix<double> m;
        public Double Scale { get; private set; } = 1;
        public static double Max;
        public ulong BlockSize { get; private set; }



        public RawMatrix(Matrix<double> m, double scale, EMatrixFormat format, ulong BlockSize)
        {
            Scale = scale;
            Format = format;
            this.m = (m * scale).PointwiseRound();
            this.BlockSize = BlockSize;
            Max = Math.Max(Max, m.Enumerate().Max(x => (x < 0) ? -x : x));
        }

        private RawMatrix() { }

        public Matrix<double> Decrypt(IComputationEnvironment env)
        {
            Max = Math.Max(Max, m.Enumerate().Max(x => (x < 0) ? -x : x));

            return m / Scale;
        }


        public static RawMatrix Read(StreamReader str)
        {
            var mtx = new RawMatrix()
            {
                Scale = Double.Parse(str.ReadLine()),
                m = DelimitedReader.Read<double>(str),
                BlockSize = ulong.Parse(str.ReadLine())
            };
            Max = Math.Max(Max, mtx.m.Enumerate().Max(x => (x < 0) ? -x : x));

            return mtx;
        }
        public void Dispose()
        {
            m = null;
        }

        public void Write(StreamWriter str)
        {
            str.WriteLine(Scale);
            DelimitedWriter.Write<double>(str, m);
            str.WriteLine(BlockSize);
        }

        public IVector Mul(IVector v, IComputationEnvironment env, bool ForceDenseFormat = false)
        {
            var vr = v as RawVector;
            var res = new RawVector(m.Multiply((Vector<double>)vr.Data), 1, v.BlockSize);
            res.RegisterScale(Scale * vr.Scale);
            return res;
        }

        public IMatrix ElementWiseMultiply(IMatrix m, IComputationEnvironment env)
        {
            if (m.Format != Format) throw new Exception("Format mismatch");
            if (m.RowCount != RowCount) throw new Exception("row-count mismatch");
            if (m.ColumnCount != ColumnCount) throw new Exception("column count mismatch");
            var mr = m as RawMatrix;
            var res = new RawMatrix(this.m.PointwiseMultiply(mr.m), 1, Format, m.BlockSize)
            {
                Scale = Scale * m.Scale
            };
            return res;

        }

        public IMatrix Add(IMatrix m, IComputationEnvironment env)
        {
            if (m.Format != Format) throw new Exception("Format mismatch");
            if (m.RowCount != RowCount) throw new Exception("Row count mismatch");
            if (m.ColumnCount != ColumnCount) throw new Exception("Column count mismatch");
            if (m.Scale != Scale) throw new Exception("Scale mismatch");
            var mr = m as RawMatrix;
            return new RawMatrix(this.m.Add(mr.m), Scale, Format, m.BlockSize);

        }

        public IVector GetColumn(int columnNumber)
        {
            if (columnNumber >= m.ColumnCount) throw new Exception("Column does not exist");
            if (Format != EMatrixFormat.ColumnMajor) throw new Exception("Columns can be extracted only from a column major matrix");
            var col = m.Column(columnNumber);
            var v = new RawVector(col, 1, BlockSize);
            v.RegisterScale(Scale);
            return v;
        }

        public IVector GetRow(int rowNumber)
        {
            if (rowNumber >= m.RowCount) throw new Exception("Row does not exist");
            if (Format != EMatrixFormat.RowMajor) throw new Exception("Row can be extracted only from a row major matrix");
            var v = new RawVector(m.Row(rowNumber), 1, BlockSize);
            v.RegisterScale(Scale);
            return v;
        }

        public void SetColumn(int columnNumber, IVector vector)
        {
            if (columnNumber >= m.ColumnCount) throw new Exception("Column does not exist");
            if (Format != EMatrixFormat.ColumnMajor) throw new Exception("Columns can be set only from a column major matrix");
            m.SetColumn(columnNumber, (Vector<double>)vector.Data);
            Max = Math.Max(Max, m.Enumerate().Max(x => (x < 0) ? -x : x));
        }



        public void RegisterScale(double scale)
        {
            Scale = scale;
        }

        public IVector ConvertToColumnVector(IComputationEnvironment env)
        {
            if ((ulong)m.ColumnCount * (ulong)m.RowCount > this.BlockSize) throw new Exception("block too long for interleaving");

            var v = new RawVector(Vector<double>.Build.DenseOfEnumerable(m.Enumerate()), 1, BlockSize);
            v.RegisterScale(Scale);
            return v;
        }

        Vector<double> Shift(Vector<double> v, int shift)
        {
            var w = Vector<double>.Build.Dense(v.Count);
            if (shift < 0)
                v.CopySubVectorTo(w, -shift, 0, v.Count + shift);
            else
                v.CopySubVectorTo(w, 0, shift, v.Count - shift);
            return w;
        }

        public IVector Interleave(int shift, IComputationEnvironment env)
        {
            int items = (shift > 0) ? shift : -shift;
            bool allignedToEnd = (shift < 0);
            if (items == 0) throw new Exception("number of items cannot be zero");
            var w = m.Column(0);
            for (int i = 1; i < m.ColumnCount; i++)
            {
                w = w.Add(Shift(m.Column(i), shift * i));
            }
            var t = new RawVector(w, 1, BlockSize);
            t.RegisterScale(Scale);
            return t;
        }

        public ulong RowCount { get { return (ulong)m.RowCount; } }
        public ulong ColumnCount { get { return (ulong)m.ColumnCount; } }

        public object Data { get { return Matrix<double>.Build.DenseOfMatrix(m); } }
        public bool IsEncrypted { get { return false; } }

        public bool DataDisposedExternaly { get; set; } = false;
    }
}
