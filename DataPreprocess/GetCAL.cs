// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

﻿using System.IO;

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Drawing;
using System.Text;

namespace DataPreprocess
{

    public class GetCAL
    {
        public static void Run(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("DataPreprocess CAL <catalog> <directory>");
                Console.WriteLine();
                Console.WriteLine("To test on a set of images, put the images in a directory and create a tab seperated catalog file with the first column being the name of the image file and a second column the index of its category.");
                Console.WriteLine();
                Console.WriteLine("To apply to the caltech dataset, download it from http://www.vision.caltech.edu/Image_Datasets/Caltech101/ and unzip it. Catalog files from the training and testing portions of the data exists here under the name cal_train.tsv and cal_test.tsv ");
                return;
            }
            var lines = File.ReadAllLines(args[0]);
            var res = lines.AsParallel().Select(l => _Run_failover(l, args[1]));
            foreach (var l in res)
            {
                Console.WriteLine(l);
            }
        }

        static string _Run_failover(string line, string imagesFolder)
        {
            string catalogName = GetTempFileNameWithExtension("tsv");
            File.WriteAllText(catalogName, line);
            string results = "Internal Error";
            var fail = false;
            try
            {
                results = _Run(catalogName, imagesFolder);
            }
            catch 
            {
                fail = true;
            }
            if (fail)
            {
                var f = line.Split('\t');
                var bmp = (Bitmap)Image.FromFile(Path.Combine(imagesFolder, f[0]));
                var bmp2 = new Bitmap(bmp.Width, bmp.Height);
                using (Graphics g = Graphics.FromImage(bmp2))
                {
                    g.DrawImage(bmp, 0, 0);
                }
                string imageName = GetTempFileNameWithExtension("jpg");
                bmp2.Save(imageName);
                File.WriteAllText(catalogName, String.Format("{0}\t{1}", imageName, f[1]));
                try
                {
                    results = _Run(catalogName, "" );
                }
                catch 
                {
                    results = "failure on line " + line;
                }
                File.Delete(imageName);
            }

            File.Delete(catalogName);
            return results;

        }

        static string GetTempFileNameWithExtension(string extension)
        {
            var tempName = Path.GetRandomFileName().Replace('.', '_') + "." + extension;
            return tempName;
        }

        static string _Run(string imagesCatalog, string imagesFolder)
        {
            var mlContext = new MLContext();

            var cols = new[] {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Label", DataKind.Int32, 1),
                };

            var data = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            {
                Columns = cols
            }).Load(imagesCatalog);

            var pipeline = mlContext.Transforms.LoadImages(imageFolder:imagesFolder, outputColumnName: "ImageReal", inputColumnName: "ImagePath")
            .Append(mlContext.Transforms.ResizeImages(outputColumnName:"ImageObject", inputColumnName:"ImageReal", imageWidth: 227, imageHeight: 227))
            .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
            .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturizedImage", m => m.ModelSelector.AlexNet(mlContext, m.OutputColumn, m.InputColumn), "Pixels"));

            

            var transformedData = pipeline.Fit(data).Transform(data);

            var Features = transformedData.GetColumn<float[]>(columnName:"FeaturizedImage").ToArray();
            var Labels = transformedData.GetColumn<Int32>(columnName: "Label").ToArray();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Labels.Length; i++)
            {
                sb.AppendFormat("{0}\t4096", Labels[i]);
                for (int j = 0; j < 4096; j++)
                    if (Features[i][j] != 0)
                        sb.AppendFormat("\t{0}:{1}", j, Features[i][j]);
            }
            return sb.ToString();



        }
    }
}
