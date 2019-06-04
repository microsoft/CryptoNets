// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using HEWrapper;

namespace NeuralNetworks
{
    public interface INetwork : IDisposable
    {
        /// <summary>
        /// applies the layer to the vector v
        /// </summary>
        /// <param name="v">the vector of number to apply the layer to</param>
        /// <returns>the results of applying the layer to v</returns>
        IMatrix Apply(IMatrix v);

        /// <summary>
        /// calls the input layer and applies this layer to the output of the input layer
        /// </summary>
        /// <returns>the results of applying the layer to the output of the input layer</returns>
        IMatrix GetNext();

        /// <summary>
        /// the dimension of the output of this layer
        /// </summary>
        /// <returns>the dimension</returns>
        int OutputDimension();

        /// <summary>
        /// returns the parent (input) of this layer
        /// </summary>
        /// <returns> the parent (input) of this layer</returns>
        INetwork GetSource();
        /// <summary>
        /// prepares this layer before first execution
        /// </summary>
        void Prepare();

        /// <summary>
        /// prepares this layer and its sources
        /// </summary>
        void PrepareNetwork();
        /// <summary>
        /// The scale of the output of the network
        /// </summary>
        double GetOutputScale();

        /// <summary>
        /// The factory in which the output of this layer is encoded
        /// </summary>
        IFactory Factory { get; set; }

        void DisposeNetwork();

        INetwork Source { get; set; }
    }

    public interface IInputLayer : INetwork
    {
        int[] Labels { get; }
    }

}

