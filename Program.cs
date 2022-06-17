// See https://aka.ms/new-console-template for more information

using ML.NET_Simple_Linear_Regression;


Console.WriteLine("Hello, ML.NET!");
Console.WriteLine("--------------");

var tests = new Tests();

//Using The Model Builder
//tests.ModelBuilderTest();


//Using The Manual Implementation
tests.ManualImplementationTest();




Console.ReadKey();