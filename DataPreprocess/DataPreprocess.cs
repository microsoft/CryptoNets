// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Linq;

namespace DataPreprocess
{
    class DataPreprocess
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Supported options are: MNIST/CIFAR/CAL");
                return;
            }
            switch (args[0])
            {
                case "MNIST": GetMNIST.Run(args.Skip(1).ToArray());
                    break;
                case "CIFAR": GetCIFAR.Run(args.Skip(1).ToArray());
                    break;
                case "CAL": GetCAL.Run(args.Skip(1).ToArray());
                    break;
                default: Console.WriteLine("Unknown option");
                    break;
            }
        }
    }
}
