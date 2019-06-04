// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System;

namespace HEWrapper
{
    /// <summary>
    /// The format of the vector
    /// </summary>
    public enum EVectorFormat { dense, sparse }

    /// <summary>
    /// A vector
    /// </summary>
    public interface IVector : IDisposable
    {
        /// <summary>
        /// Decrypt a vector
        /// </summary>
        /// <param name="env">The computational environment that includes the secret keys</param>
        /// <returns>Decrypted value</returns>
        Vector<double> Decrypt(IComputationEnvironment env);
        /// <summary>
        /// Decrypt a vector while ignoring scale such that the full precision is kept
        /// </summary>
        /// <param name="env">The computational environment that includes the secret keys</param>
        /// <returns>Decrypted values</returns>
        IEnumerable<BigInteger> DecryptFullPrecision(IComputationEnvironment env);
        /// <summary>
        /// Write a vector to a stream
        /// </summary>
        /// <param name="str">The stream to write to</param>
        void Write(StreamWriter str);
        /// <summary>
        /// The underlying representation of the vector. Only for advance usage.
        /// </summary>
        object Data { get; }
        /// <summary>
        /// Subtract vectors
        /// </summary>
        /// <param name="v">The vector to subtract from the current vector</param>
        /// <param name="env">The computatonal environment</param>
        /// <returns>The result of the subtraction</returns>
        IVector Subtract(IVector v, IComputationEnvironment env);
        /// <summary>
        /// Add vectors
        /// </summary>
        /// <param name="v">The vector to add to the current vector</param>
        /// <param name="env">The computational environment</param>
        /// <returns>The result of the addition</returns>
        IVector Add(IVector v, IComputationEnvironment env);
        /// <summary>
        /// Compute the dot-product of two vectors
        /// </summary>
        /// <param name="v">The vector to multiply by</param>
        /// <param name="env">The computational environment</param>
        /// <returns>The result of the inner product as a vector with a single element</returns>
        IVector DotProduct(IVector v, IComputationEnvironment env);
        /// <summary>
        /// Compute the dot-product of length elements
        /// </summary>
        /// <param name="v">The vector to multiply by</param>
        /// <param name="length">The number of elements to use for dot-product</param>
        /// <param name="env">The computational environment</param>
        /// <returns>Vector of results</returns>
        IVector DotProduct(IVector v, ulong length, IComputationEnvironment env);
        /// <summary>
        /// Multiply two vectors element-wise
        /// </summary>
        /// <param name="v">The vector to multiply by</param>
        /// <param name="env">The computational environment</param>
        /// <returns>Vector of results</returns>
        IVector PointwiseMultiply(IVector v, IComputationEnvironment env);
        /// <summary>
        /// Sum all the elements of a vector
        /// </summary>
        /// <param name="env">The computational environment</param>
        /// <returns>A vector containing a single element which is the sum</returns>
        IVector SumAllSlots(IComputationEnvironment env);
        /// <summary>
        /// Createa new vector that contains several duplications of the current vector
        /// </summary>
        /// <param name="count">The number of replications</param>
        /// <param name="env">The computational environment</param>
        /// <returns>The new vector with duplications of the current vector</returns>
        IVector Duplicate(ulong count, IComputationEnvironment env);
        /// <summary>
        /// Rotate the elements of the vector. The i'th element will be placed at (i + amount) mod blockSize 
        /// </summary>
        /// <param name="amount">The size rotation </param>
        /// <param name="env">The computational environment</param>
        /// <returns>Rotated vector</returns>
        IVector Rotate(int amount, IComputationEnvironment env);
        /// <summary>
        /// Create a permutation of the elements of the vector. The permutation is performed by selecting elements and rotating each selection by a certain amount and adding all the results of this selection-rotation process.
        /// </summary>
        /// <param name="selections">a vectors of selections of the elements</param>
        /// <param name="shifts">sizes of rotations</param>
        /// <param name="outputDim">Desired dimension of the output vector</param>
        /// <param name="env">The computational environment</param>
        /// <returns>Perumted vector</returns>
        IVector Permute(IVector[] selections, int[] shifts, ulong outputDim, IComputationEnvironment env);
        /// <summary>
        /// The dimension of the vector
        /// </summary>
        ulong Dim { get; }
        /// <summary>
        /// The scale of the vector
        /// </summary>
        double Scale { get;}
        /// <summary>
        /// Force the scale of the vector 
        /// </summary>
        /// <param name="scale">New scale</param>
        void RegisterScale(double scale);
        /// <summary>
        /// Returns true if the values of the vector are encrypted and false otherwise
        /// </summary>
        bool IsEncrypted { get; }
        /// <summary>
        /// Returns true if the vector encodes signed integers and flase if unsigned integers are encoded
        /// </summary>
        bool IsSigned { get; set; }
        /// <summary>
        /// The underlying block-size of the vectors
        /// </summary>
        ulong BlockSize { get; }
        // The vector format
        EVectorFormat Format { get; }

    }
}
