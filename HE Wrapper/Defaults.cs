// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;

namespace HEWrapper
{
    public static class Defaults
    {
        public static IFactory RawFactory { get; } = new RawFactory(8192);
        static int _threadCount = Environment.ProcessorCount;
        /// <summary>
        /// number of threads to use for parallel execution
        /// </summary>
        static public int ThreadCount { get { return _threadCount;} set { _threadCount = value; } }
    }
}
