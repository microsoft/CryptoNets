// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System;
using System.Numerics;
using System.Security.Cryptography;

namespace HEWrapper
{
    static public class Randomness
    {
        static RNGCryptoServiceProvider CryptoRNG = new RNGCryptoServiceProvider();
        [ThreadStatic]
        static byte[] Buffer;


        public static UInt64 GetIntInRange(UInt64 max)
        {
            if (Buffer == null || Buffer.Length < 8)
            {
                Buffer = new byte[8];
            }

            CryptoRNG.GetBytes(Buffer, 0, 8);
            UInt64 temp = BitConverter.ToUInt64(Buffer, 0);
            return (UInt64)(max * (temp / (UInt64.MaxValue + 1.0)));
        }

        public static UInt64 GetIntInRange(UInt64 min, UInt64 max)
        {
            return min + GetIntInRange(max - min);
        }


        public static BigInteger GetIntInRange(BigInteger max)
        {
            var maxArray = max.ToByteArray();
            var l = maxArray.Length;
            if (Buffer == null || Buffer.Length < l)
            {
                Buffer = new byte[l];
            }

            while(true)
            {
                CryptoRNG.GetBytes(Buffer, 0, l);
                for (int i = l - 1; i >= 0; i++)
                    if (Buffer[i] < maxArray[i])
                        return new BigInteger(Buffer);

            }
        }
        public static BigInteger GetIntInRange(BigInteger min, BigInteger max)
        {
            return min + GetIntInRange(max - min);
        }
    }
}
