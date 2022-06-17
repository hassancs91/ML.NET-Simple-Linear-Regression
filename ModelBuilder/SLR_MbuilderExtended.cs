using Microsoft.ML;


namespace ML.NET_Simple_Linear_Regression.ModelBuilder
{
    public class SLR_MbuilderExtended
    {
        public string TrainedModelPath { get; set; }


        public SLR_MbuilderExtended(string trainedModelPath)
        {
            TrainedModelPath = trainedModelPath;
        }


        //public SLR_MBuilder.ModelOutput Predict(Single years)
        //{
        //    // Create single instance of sample data from first line of dataset for model input
        //    SLR_MBuilder.ModelInput sampleData = new SLR_MBuilder.ModelInput()
        //    {
        //        YearsExperience = Convert.ToSingle(years),
        //    };

        //    // Make a single prediction on the sample data and print results
        //    var predictionResult = SLR_MBuilder.Predict(sampleData);

        //    return predictionResult;

        //}


        public SLR_MBuilder.ModelOutput Predict(Single years, string MLNetModelPath)
        {
            var PredictEngine =
                new Lazy<PredictionEngine<SLR_MBuilder.ModelInput, SLR_MBuilder.ModelOutput>>(() => CreatePredictEngine(MLNetModelPath), true);

            var sampleData = new SLR_MBuilder.ModelInput()
            {
                YearsExperience = Convert.ToSingle(years),
            };


            var predictEngine = PredictEngine.Value;
            return predictEngine.Predict(sampleData);
        }


        private PredictionEngine<SLR_MBuilder.ModelInput, SLR_MBuilder.ModelOutput> CreatePredictEngine(string MLNetModelPath)
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<SLR_MBuilder.ModelInput, SLR_MBuilder.ModelOutput>(mlModel);
        }



        public string ReTrainModel(string newCsvDataPath, string NewModelOutPutPath)
        {
            // Create MLContext
            var mlContext = new MLContext();


            //Load Data
            var newData = mlContext.Data.LoadFromTextFile<SLR_MBuilder.ModelInput>(newCsvDataPath,hasHeader:true,separatorChar: ',');

            
            var newModel = SLR_MBuilder.RetrainPipeline(mlContext, newData);

            //Save New Model
            var datePostFix = DateTime.Now.ToString("_MMddyyyy_HHmm");
            string outPutModelPath = NewModelOutPutPath + "SLR" + datePostFix + ".zip";
            mlContext.Model.Save(newModel, newData.Schema, outPutModelPath);

            return outPutModelPath;
        }


    }
}
