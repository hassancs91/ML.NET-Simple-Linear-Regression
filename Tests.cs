using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.NET_Simple_Linear_Regression.Manual_Implementation;
using ML.NET_Simple_Linear_Regression.ModelBuilder;

namespace ML.NET_Simple_Linear_Regression
{
    internal class Tests
    {

        public void ModelBuilderTest()
        {
            Console.WriteLine("Test Prediction With Model Builder.");
            Console.WriteLine("Please Enter Number of Years: ");
            var years = Console.ReadLine();
            var sampleData = new SLR_MBuilder.ModelInput()
            {
                YearsExperience = Convert.ToSingle(years),
            };

            //Load model and predict output
            var result = SLR_MBuilder.Predict(sampleData);
            Console.WriteLine("Predicted Salary: " + result.Score);
            Console.WriteLine("-----------------------------------");


            Console.WriteLine("Retrain Data Test - Based Implementation");

            //Path To Save New Trained Models
            var newOutPutPath =
                @"E:\LearnWithHasan\MachineLearning\ML.NET Simple Linear Regression\ModelBuilder\";

            //Get This Path From SLR_MBuilder.consumption.cs
            var currentModelPath =
                @"E:\LearnWithHasan\MachineLearning\ML.NET Simple Linear Regression\ModelBuilder\SLR_MBuilder.zip";

            //Path To The New Data File on your PC.
            var newCsvDataFilePath =
                @"E:\LearnWithHasan\MachineLearning\DataSets\SLR\retrain.csv";

            //Create Instance Our Extended Class
            var slrExtended = new SLR_MbuilderExtended(currentModelPath);

            //Call The Retrain Method
            slrExtended.ReTrainModel(newCsvDataFilePath, newOutPutPath);


            Console.WriteLine("-------------Retraining Completed--------------\n");


            Console.WriteLine("Test Prediction With New Trained Model.");
            Console.WriteLine("Please Enter Number of Years: ");
            years = Console.ReadLine();
            Console.WriteLine("Please Enter Model Name: ");
            var modelName = Console.ReadLine();
            //Call The Predict Method
            var newResult = slrExtended.Predict(Convert.ToSingle(years), newOutPutPath + modelName);
            Console.WriteLine("New Predicted Salary: " + newResult.Score);
            Console.WriteLine("-----------------------------------");
        }

        public void ManualImplementationTest()
        {
            var slrManual = new SLR_ManualModel();
            slrManual.BuildAndTestModel();
        }

    }
}
