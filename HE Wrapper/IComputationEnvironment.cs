// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿namespace HEWrapper
{
    /// <summary>
    /// ComputationalEnvironments are used by each scheme to store information that is needed
    /// for performing operations on encrypted data. An environment is not expected to be used 
    /// in multiple threads at the same time and therefore each thread should hold its own 
    /// computational environment.
    /// </summary>
    public interface IComputationEnvironment
    {
        /// <summary>
        /// the factory that was used to create this environment
        /// </summary>
        IFactory ParentFactory { get; }

        /// <summary>
        /// the prime factors used as plaintext modulii
        /// </summary>
        ulong[] Primes { get; }

    }
}
