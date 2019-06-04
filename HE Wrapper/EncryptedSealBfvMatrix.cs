// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace HEWrapper
{
    /// <summary>
    /// represents an encrypted matrix using SEAL
    /// </summary>
    public class EncryptedSealBfvMatrix : IMatrix
    {
        EncryptedSealBfvVector[] leVectors;

        public ulong RowCount { get { if (Format == EMatrixFormat.RowMajor) return (ulong)leVectors.Length; else return leVectors[0].Dim; } }

        public ulong ColumnCount { get { if (Format == EMatrixFormat.ColumnMajor) return (ulong)leVectors.Length; else return leVectors[0].Dim; } }
        public object Data { get { return leVectors; } }

        public double Scale { get { return leVectors[0].Scale; } }
        public EMatrixFormat Format { get; set; } = EMatrixFormat.ColumnMajor;

        public ulong BlockSize { get { return leVectors[0].BlockSize; } }
#if DEBUG
        readonly string Trace = Environment.StackTrace;
#endif

        public EncryptedSealBfvMatrix(EncryptedSealBfvVector[] columns, IComputationEnvironment env, bool CopyVectors = true)
        {
            if (columns != null && columns.Any(c => c.Dim != columns[0].Dim))
                throw new Exception("all columns of a matrix should have the same size");
            if (columns != null)
            {
                leVectors = (CopyVectors) ? columns.Select(x => new EncryptedSealBfvVector(x, env)).ToArray() : columns;
            }
        }

        public EncryptedSealBfvMatrix()
        {
        }
#if DEBUG
        ~EncryptedSealBfvMatrix()
        {
            if (leVectors != null && !DataDisposedExternaly)
                throw new Exception(String.Format("Data that was allocated in the following context was not disposed:\n{0}", Trace));
        }
#endif

        public void Dispose()
        {
            if (leVectors != null && !DataDisposedExternaly)
                foreach (var v in leVectors)
                    if (v != null)
                        v.Dispose();
            leVectors = null;
        }
        public Matrix<double> Decrypt(IComputationEnvironment env)
        {
            var leenv = env as EncryptedSealBfvEnvironment;
            var vectors = new Vector<double>[leVectors.Length];
            Utils.ParallelProcessInEnv(vectors.Length, env, (penv, taskIndex, k) =>
              vectors[k] = leVectors[k].Decrypt(penv));
            Matrix<double> res = (Format == EMatrixFormat.RowMajor) ? Matrix<double>.Build.DenseOfRowVectors(vectors) : Matrix<double>.Build.DenseOfColumnVectors(vectors);
            return res;
        }

        public IVector Mul(IVector v, IComputationEnvironment env, bool ForceDenseFormat = false)
        {
            var ev = v as EncryptedSealBfvVector;
            var lenv = env as EncryptedSealBfvEnvironment;
            if (Format == EMatrixFormat.ColumnMajor)
            {
                if (ForceDenseFormat) throw new Exception("Forcing dense format is available only in RowMajor mode");
                return EncryptedSealBfvVector.DenseMatrixBySparseVectorMultiply(leVectors, ev, lenv);
            }
            if (!ForceDenseFormat)
            {
                IVector[] tempVectors = new EncryptedSealBfvVector[leVectors.Length];
                Utils.ParallelProcessInEnv(leVectors.Length, env, (penv, task, colIndex) =>
                {
                    tempVectors[colIndex] = leVectors[colIndex].DotProduct(ev, penv);
                });
                var res = EncryptedSealBfvVector.GenerateSpareOfArray(tempVectors, env);
                foreach (var t in tempVectors) t.Dispose();
                return res;
            }
            // Column major mode forcing dense output;
            {
                var sumVectors = new EncryptedSealBfvVector[Defaults.ThreadCount];
                Utils.ParallelProcessInEnv(leVectors.Length, env, (penv, task, colIndex) =>
                {
                    var t = (EncryptedSealBfvVector)leVectors[colIndex].DotProduct(ev, penv, ForceOutputInColumn: colIndex);
                    var s = sumVectors[task];
                    sumVectors[task] = (sumVectors[task] == null) ? t : (EncryptedSealBfvVector)sumVectors[task].Add(t, env);
                    if (sumVectors[task] != t) t.Dispose();
                    if (s != null) s.Dispose();
                });
                EncryptedSealBfvVector sum = null;

                for (int i = 0; i < sumVectors.Length; i++)
                {
                    if (sumVectors[i] == null) continue;
                    if (sum == null)
                        sum = sumVectors[i];
                    else
                    {
                        var t = sum;
                        sum = (EncryptedSealBfvVector)sum.Add(sumVectors[i], env);
                        t.Dispose();
                        sumVectors[i].Dispose();
                    }
                }
                sum.RegisterDim((ulong)leVectors.Length);
                if (sum.Format != EVectorFormat.dense)
                    throw new Exception("Internal probloem: expecting the output to be dense");
                return sum;
            }
        }

        public IMatrix Add(IMatrix m, IComputationEnvironment env)
        {
            if (m.Format != Format) throw new Exception("Format mismatch");
            if (m.RowCount != RowCount) throw new Exception("Row count mismatch");
            if (m.ColumnCount != ColumnCount) throw new Exception("Column count mismatch");
            var me = m as EncryptedSealBfvMatrix;
            var lenv = env as EncryptedSealBfvEnvironment;
            var tempVectors = new EncryptedSealBfvVector[leVectors.Length];
            Utils.ParallelProcessInEnv(leVectors.Length, (penv, task, colIndex) =>
            {
                tempVectors[colIndex] = (EncryptedSealBfvVector)leVectors[colIndex].Add(me.leVectors[colIndex], penv);
            }, lenv.ParentFactory);
            return new EncryptedSealBfvMatrix() { leVectors = tempVectors, Format = Format};

        }


        public IMatrix ElementWiseMultiply(IMatrix m, IComputationEnvironment env)
        {
            if (m.Format != Format) throw new Exception("Format mismatch");
            if (m.RowCount != RowCount) throw new Exception("Row-count mismatch");
            if (m.ColumnCount != ColumnCount) throw new Exception("Column count mismatch");
            var me = m as EncryptedSealBfvMatrix;
            var lenv = env as EncryptedSealBfvEnvironment;
            var tempVectors = new EncryptedSealBfvVector[leVectors.Length];

            Utils.ParallelProcessInEnv(leVectors.Length, lenv, (penv, task, colIndex) =>
            {
                tempVectors[colIndex] = (EncryptedSealBfvVector)leVectors[colIndex].PointwiseMultiply(me.leVectors[colIndex], penv);
            });
            return new EncryptedSealBfvMatrix() { leVectors = tempVectors, Format = Format};
        }

        public IVector GetColumn(int columnNumber)
        {
            if (columnNumber >= leVectors.Length) throw new Exception("Column does not exist");
            if (Format != EMatrixFormat.ColumnMajor) throw new Exception("Columns can be extracted only from a column major matrix");
            return leVectors[columnNumber];
        }

        public IVector GetRow(int rowNumber)
        {
            if (rowNumber >= leVectors.Length) throw new Exception("Row does not exist");
            if (Format != EMatrixFormat.RowMajor) throw new Exception("Rows can be extracted only from a row major matrix");
            return leVectors[rowNumber];
        }

        public void SetColumn(int columnNumber, IVector vector)
        {
            if (columnNumber >= leVectors.Length) throw new Exception("Column does not exist");
            if (Format != EMatrixFormat.ColumnMajor) throw new Exception("Columns can be set only from a column major matrix");
            if (vector.Dim != leVectors[columnNumber].Dim) throw new Exception("dimension of vector does not match the dimension of the vector it is replacing");
            if (vector.Scale != leVectors[columnNumber].Scale) throw new Exception("Scale of vector does not match the scale of the vector it is replacing");
            if (vector.IsEncrypted != leVectors[columnNumber].IsEncrypted) throw new Exception("can't exchange encrypted and not encrypted vectors");
            if (!(vector is EncryptedSealBfvVector v)) throw new Exception("expecting LargeEncryptedVector");
            leVectors[columnNumber] = v;
        }


        public static EncryptedSealBfvMatrix Read(StreamReader str, EncryptedSealBfvEnvironment env)
        {
            var mtx = new EncryptedSealBfvMatrix();
            var line = str.ReadLine();
            if (line != "<Start LargeEncryptedMatrix>") throw new Exception("Bad stream format.");
            mtx.Format = (EMatrixFormat)Enum.Parse(mtx.Format.GetType(), str.ReadLine());
            mtx.leVectors = new EncryptedSealBfvVector[int.Parse(str.ReadLine())];
            for (int i = 0; i < mtx.leVectors.Length; i++)
            {
                mtx.leVectors[i] = EncryptedSealBfvVector.Read(str, env);
            }

            line = str.ReadLine();
            if (line != "<End LargeEncryptedMatrix>") throw new Exception("Bad stream format.");
            return mtx;
        }

        public void Write(StreamWriter str)
        {
            str.WriteLine("<Start LargeEncryptedMatrix>");
            str.WriteLine(Enum.GetName(Format.GetType(), Format));
            str.WriteLine(leVectors.Length);
            for (int i = 0; i < leVectors.Length; i++)
                leVectors[i].Write(str);
            str.WriteLine("<End LargeEncryptedMatrix>");
            str.Flush();
        }

        public void RegisterScale(double scale)
        {
            foreach (var v in leVectors) v.RegisterScale(scale);
        }

        public IVector ConvertToColumnVector(IComputationEnvironment env)
        {
            var lenv = env as EncryptedSealBfvEnvironment;
            return EncryptedSealBfvVector.Stack(leVectors, lenv);
        }

        public IVector Interleave(int shift, IComputationEnvironment env)
        {
            if (this.Format != EMatrixFormat.ColumnMajor) throw new Exception("Expecting ColumnMajor matrix");
            return EncryptedSealBfvVector.Interleave(leVectors, shift, (EncryptedSealBfvEnvironment) env);
            
        }

        public bool IsEncrypted { get { return leVectors.All(v => v.IsEncrypted); } }

        public bool DataDisposedExternaly { get; set; } = false;
    }
}
