using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using ML.NET_Simple_Linear_Regression.ModelBuilder;

namespace ML.NET_Simple_Linear_Regression.Manual_Implementation
{
    internal class SLR_ManualModel
    {

        public void BuildAndTestModel()
        {
            var context = new MLContext();


            var CsvDataFilePath =
                @"E:\LearnWithHasan\MachineLearning\DataSets\SLR\salaries.csv";


            var data = context.Data.LoadFromTextFile<ModelInput>(CsvDataFilePath, hasHeader: true,
                separatorChar: ',');

            var pipeline = context.Transforms.ReplaceMissingValues(@"YearsExperience", @"YearsExperience")
                .Append(context.Transforms.Concatenate(@"Features", @"YearsExperience"))
                .Append(context.Transforms.NormalizeMinMax(@"Features", @"Features"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(data);

            var predictions = model.Transform(data);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");

            context.Model.Save(model, data.Schema,"model.zip");

            //Try Prediction

            var PredictEngine =
                new Lazy<PredictionEngine<ModelInput,ModelOutput>>(() => CreatePredictEngine("model.zip"), true);



            Console.WriteLine("Test Prediction With Model Builder.");
            Console.WriteLine("Please Enter Number of Years: ");
            var years = Console.ReadLine();
            var sampleData = new ModelInput()
            {
                YearsExperience = Convert.ToSingle(years),
            };


            var predictEngine = PredictEngine.Value;
            var result = predictEngine.Predict(sampleData);

            Console.WriteLine("Predicted Salary: " + result.Score);
            Console.WriteLine("-----------------------------------");

        }


        private PredictionEngine<ModelInput,ModelOutput> CreatePredictEngine(string MLNetModelPath)
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }
        public class ModelInput
        {
            [ColumnName(@"YearsExperience"), LoadColumn(0)]
            public float YearsExperience { get; set; }


            [ColumnName(@"Label"), LoadColumn(1)]
            public float Salary { get; set; }

        }


        public class ModelOutput
        {
            public float Score { get; set; }
        }

    }





}
