// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System;

namespace HEWrapper
{
    /// <summary>
    /// ColumnMajor matrices are stored as array of vectors such that each vector is a column of the matrix
    /// RowMajor matrices are stores as array of vectors such that each vector is a row of the matrix
    /// </summary>
    public enum EMatrixFormat {ColumnMajor, RowMajor };
    /// <summary>
    /// Represents a matrix
    /// </summary>
    public interface IMatrix : IDisposable
    {
        /// <summary>
        /// Decrypt a matrix
        /// </summary>
        /// <param name="env"> the computational environment to use for decryption. The environment shold contain the secret key to allow decryption</param>
        /// <returns></returns>
        Matrix<double> Decrypt(IComputationEnvironment env);
        /// <summary>
        /// Write the matrix to a stream
        /// </summary>
        /// <param name="str">an pen stream to write the matrix to</param>
        void Write(StreamWriter str);
        /// <summary>
        /// Multiply a matrix by a vector. For the matrix M and the vector v it computes Mv. The implementation of the multiplication method may depend on the representation used for the matrix and the vector
        /// </summary>
        /// <param name="v">the vector to multiply the matrix by</param>
        /// <param name="env"> computational environment</param>
        /// <param name="ForceDenseFormat">When multiplying a row major matrix by a dense vector the result may be in sparse format. By setting this parameter the result will be converted to dense format</param>
        /// <returns>vector of the result</returns>
        IVector Mul(IVector v, IComputationEnvironment env, bool ForceDenseFormat = false);
        /// <summary>
        /// add to matrices in the same format
        /// </summary>
        /// <param name="m">the matrix to add</param>
        /// <param name="env">the computational environment</param>
        /// <returns>matrix of result</returns>
        IMatrix Add(IMatrix m, IComputationEnvironment env);
        /// <summary>
        /// Multiply two matrices elementwise. Matrices should have the same representation
        /// </summary>
        /// <param name="m">the matrix to multiply by</param>
        /// <param name="env">the computational environment</param>
        /// <returns>matrix of results</returns>
        IMatrix ElementWiseMultiply(IMatrix m, IComputationEnvironment env);
        /// <summary>
        /// The number of rows in the matrix
        /// </summary>
        ulong RowCount { get; }
        /// <summary>
        /// The number of columns in the matrix
        /// </summary>
        ulong ColumnCount { get; }
        /// <summary>
        /// The internal representation of the data. Only for advance usage.
        /// </summary>
        object Data { get; }
        /// <summary>
        /// Scale is the precision of the values. Each value is multiplied by scale beore rounded to an integer
        /// </summary>
        double Scale { get; }
        /// <summary>
        /// Force the scale of the matrix without changing its values. Only for advance usage.
        /// </summary>
        /// <param name="scale">New scale</param>
        void RegisterScale(double scale);
        /// <summary>
        /// The matrix format.
        /// </summary>
        EMatrixFormat Format { get; }
        /// <summary>
        /// Return a column of a matrix if the matrix is column major.
        /// </summary>
        /// <param name="columnNumber"> The index of the column</param>
        /// <returns>The column vector.</returns>
        IVector GetColumn(int columnNumber);
        /// <summary>
        /// Return a row of a matrix if the matrix is row major.
        /// </summary>
        /// <param name="rowNumber"> The index of the row</param>
        /// <returns></returns>
        IVector GetRow(int rowNumber);
        /// <summary>
        /// Forces a column of the matrix for column majour matrix
        /// </summary>
        /// <param name="columnNumber">The index of the column</param>
        /// <param name="vector">The new vector column.</param>
        void SetColumn(int columnNumber, IVector vector);
        /// <summary>
        /// Returns true of the matrix contains encrypted values and false if all the values are plain
        /// </summary>
        bool IsEncrypted { get; }
        /// <summary>
        /// The underlying blocksize of each column/row vector
        /// </summary>
        ulong BlockSize { get; }
        /// <summary>
        /// Set this to true if the column/row vectors should not be disposed when the matrix is disposed.
        /// </summary>
        bool DataDisposedExternaly { get; set; }
        /// <summary>
        /// Converts the matrix to a single column vector
        /// </summary>
        /// <param name="env">Computational environment</param>
        /// <returns>A vector representation of the matrix</returns>
        IVector ConvertToColumnVector(IComputationEnvironment env);
        /// <summary>
        /// Creates a vector which includes interleaving items from the vectors 
        /// making up this matrix. NOTE: the matrix is assumed to have zero at every 
        /// item which is not relevant.
        /// </summary>
        /// <param name="shift"> the shift to apply</param>
        /// <returns></returns>
        IVector Interleave(int shift, IComputationEnvironment env);
    }
}
