// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.IO;
using System.Collections;

namespace NeuralNetworks
{
    public class WeightsReader
    {
        public ArrayList Weights;

        public ArrayList Biases;

        public WeightsReader(string weightsCsvPath, string biasesCsvPath)
        {
            Weights = new ArrayList();
            Biases = new ArrayList();
            ReadValuesToArray(weightsCsvPath, Weights);
            ReadValuesToArray(biasesCsvPath, Biases);
        }

        private void ReadValuesToArray(string csvPath, ArrayList arr)
        {
            using (var reader = new StreamReader(csvPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    double[] values = Array.ConvertAll(line.Split(','), Double.Parse);
                    arr.Add(values);
                }
            }
        }
    }
}
