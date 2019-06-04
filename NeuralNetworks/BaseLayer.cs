// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using HEWrapper;

namespace NeuralNetworks
{
    public abstract class BaseLayer : INetwork
    {
        internal bool layerPrepared = false;
        public INetwork Source { get; set; }

        IFactory _factory = null;

        public IFactory Factory { get { return _factory ?? Source.Factory; } set { _factory = value; } }
        abstract public IMatrix Apply(IMatrix m);

        public bool Verbose { get; set; } = false;
#if DEBUG
        public static string Trace { get; set; }
#endif
        public virtual IMatrix GetNext()
        {
            if (!layerPrepared) Prepare();
            var m = Source.GetNext();
#if DEBUG
            Trace = Environment.StackTrace;
#endif
            if (Verbose)
            {
                OperationsCount.Reset();
                DateTime start = DateTime.Now;
                var res = Apply(m);
                var end = DateTime.Now;
                Console.WriteLine("Layer {0} computed in {1} seconds ({2} -- {3}) layer width ({4},{5})", this.GetType().Name, (end - start).TotalSeconds, start.ToString("hh:mm:ss.fff"), end.ToString("hh:mm:ss.fff"), m.RowCount, m.ColumnCount);
                CryptoTracker.TestBudget(res.GetColumn(0), Factory);

                OperationsCount.Print();
                if (res != m) m.Dispose();
                return res;
            }
            else
            {
                var res = Apply(m);
                if (res != m) m.Dispose();
                return res;
            }
        }
        public virtual double GetOutputScale()
        {
            return Source.GetOutputScale();
        }

        public virtual INetwork GetSource()
        {
            return Source;
        }

        public virtual int OutputDimension()
        {
            return Source.OutputDimension();
        }

        public virtual void Prepare()
        {
        }

        public virtual void PrepareNetwork()
        {
            if (Source != null) Source.PrepareNetwork();
            if (Verbose)
            {
                OperationsCount.Reset();
                DateTime start = DateTime.Now;
                Prepare();
                layerPrepared = true;
                var end = DateTime.Now;
                Console.WriteLine("Prepare {0} computed in {1} seconds ({2} -- {3})", this.GetType().Name, (end - start).TotalSeconds, start.ToString("hh:mm:ss.fff"), end.ToString("hh:mm:ss.fff"));
                OperationsCount.Print();
            } 
            else
            {
                Prepare();
                layerPrepared = true;
            }

        }

        public void DisposeNetwork()
        {
            if (Source != null)
                Source.DisposeNetwork();
            Dispose();
        }
        public virtual void Dispose()
        { }

        protected void ProcessInEnv(Action<IComputationEnvironment> lambda) => Utils.ProcessInEnv(lambda, Factory);
        protected IVector ProcessInEnv(Func<IComputationEnvironment, IVector> lambda) => Utils.ProcessInEnv(lambda, Factory);
        protected IMatrix ProcessInEnv(Func<IComputationEnvironment, IMatrix> lambda) => Utils.ProcessInEnv(lambda, Factory);
        protected void ParallelProcessInEnv(int count, Action<IComputationEnvironment, int, int> lambda) => Utils.ParallelProcessInEnv(count, lambda, Factory);
        protected void ParallelProcessInEnv(int count, IComputationEnvironment masterEnv, Action<IComputationEnvironment, int, int> lambda) => Utils.ParallelProcessInEnv(count, masterEnv, Factory, lambda);

    }
}
