// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace HEWrapper
{
    public static class Utils
    {

        public static void ProcessInEnv(Action<IComputationEnvironment> lambda, IFactory factory)
        {
            var env = factory.AllocateComputationEnv();
            lambda(env);
            factory.FreeComputationEnv(env);
        }

        public static IVector ProcessInEnv(Func<IComputationEnvironment, IVector> lambda, IFactory factory)
        {
            var env = factory.AllocateComputationEnv();
            var res = lambda(env);
            factory.FreeComputationEnv(env);
            return res;
        }

        public static IMatrix ProcessInEnv(Func<IComputationEnvironment, IMatrix> lambda, IFactory factory)
        {
            var env = factory.AllocateComputationEnv();
            var res = lambda(env);
            factory.FreeComputationEnv(env);
            return res;
        }

        public static void ParallelProcessInEnv(int count, Action<IComputationEnvironment, int, int> lambda, IFactory factory)
        {
            ParallelProcessInEnv(count, null, factory, lambda);
        }

        public static void ParallelProcessInEnv(int count, IComputationEnvironment masterEnv, Action<IComputationEnvironment, int, int> lambda) => ParallelProcessInEnv(count, masterEnv, masterEnv.ParentFactory, lambda);

        public static void ParallelProcessInEnv(int count, IComputationEnvironment masterEnv, IFactory factory, Action<IComputationEnvironment, int, int> lambda)
        {
            if (count < 2)
            {
                int taskID = 0;
                if (masterEnv != null)
                {
                    for (int k = 0; k < count; k++)
                        lambda(masterEnv, taskID, k);
                } else
                {
                    ProcessInEnv((env) =>
                    {
                        for (int k = 0; k < count; k++)
                            lambda(env, taskID, k);
                    }, factory);
                }
            }
            else
            {
                int nextItem = -1;
                int threadCount = (Defaults.ThreadCount > count) ? count : Defaults.ThreadCount;
                Task[] tasks = new Task[threadCount];
                for (int tsk = 0; tsk < tasks.Length; tsk++)
                {
                    int taskIndex = tsk;
                    tasks[tsk] = Task.Run(() =>
                    {
                        int currentTask = (int)Task.CurrentId;
                        var env = factory.AllocateComputationEnv();
                        int k = 0;
                        while (k < count)
                        {
                            k = Interlocked.Increment(ref nextItem);
                            if (k >= count) break;
                            lambda(env, taskIndex, k);
                        }
                        factory.FreeComputationEnv(env);
                    });
                }
                Task.WaitAll(tasks);
            }
        }

        static public void Time(string name, Action lambda)
        {
            var start = DateTime.Now;
            lambda();
            var stop = DateTime.Now;
            Console.WriteLine("Time for {0}: {1}", name, (stop - start).TotalMilliseconds);
        }
        static public string RowToString(Vector<double> row)
        {
            return String.Join("\t", row.Select(x => x.ToString("N4")));
        }

        static public void Dump(string fileName, IMatrix m, IFactory factory)
        {
            Utils.ProcessInEnv(env =>
            {
                var data = m.Decrypt(env);
                var lines = data.EnumerateRows().Select(row => RowToString(row));
                File.WriteAllLines(fileName, lines);

            }, factory);
        }

        static public void Show(IMatrix m, IFactory factory)
        {
            Utils.ProcessInEnv(env =>
            {
                var data = m.Decrypt(env);
                var lines = data.EnumerateRows().Select(row => RowToString(row));
                foreach (var l in lines)
                    Console.WriteLine(l);
            }, factory);
        }
    }
}
