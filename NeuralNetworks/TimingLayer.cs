// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Collections.Generic;
using System.Text;
using HEWrapper;

namespace NeuralNetworks
{
    /// <summary>
    /// The timing layer is used to measure the time it takes to compute different parts of the neural network.
    /// The layer is dummy in the sense that it does not perform any computation by itself. 
    /// </summary>
    public class TimingLayer : BaseLayer
    {
        static readonly Dictionary<string, double> TotalTimeMS = new Dictionary<string, double>();
        static readonly Dictionary<string, int> N = new Dictionary<string, int>();
        static readonly Dictionary<string, DateTime> StartTime = new Dictionary<string, DateTime>();

        public string[] StartCounters { get; set; } = new string[0];
        public string[] StopCounters { get; set; } = new string[0];

        public static string GetStats(bool multiLines = false)
        {
            StringBuilder sb = new StringBuilder();
            bool first = true;
            foreach (var kv in TotalTimeMS)
            {
                if (!first)
                    sb.Append(multiLines ? "\n" : "\t");
                         
                first = false;
                sb.AppendFormat("{0} {1:0.00}", kv.Key, kv.Value / N[kv.Key]);
            }
            return sb.ToString();
        }

        public static void Reset()
        {
            TotalTimeMS.Clear();
            N.Clear();
            StartTime.Clear();
        }


        public override IMatrix Apply(IMatrix m)
        {
            var now = DateTime.Now;
            foreach (var c in StartCounters)
                StartTime[c] = now;
            foreach (var c in StopCounters)
            {
                if (StartTime.ContainsKey(c))
                {
                    TotalTimeMS.TryGetValue(c, out double sum);
                    TotalTimeMS[c] = sum + (now - StartTime[c]).TotalMilliseconds;
                    N.TryGetValue(c, out int n);
                    N[c] = n + 1;
                }
            }


            return m;
        }
    }
}
