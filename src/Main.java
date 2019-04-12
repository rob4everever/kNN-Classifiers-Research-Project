/**
 * Author:  100086865
 *
 * Entry point for the classifier. Data is provided to the classifier here
 * as well as showcasing the results.
 */

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Main {

    public static void main(String[]args){

        System.out.println("\nKNN Classifier ---");

        //Load the data sets
        Instances trainingData = ClassifierTools.loadClassifierData("res/Pitcher_Training.arff");
        Instances testData = ClassifierTools.loadClassifierData("res/Pitcher_Test.arff");

        //Create the classifier
        Classifier knn = new KNN(3, false, false,false);

        try {

            //Build the classifier with the training data
            knn.buildClassifier(trainingData);

            //Print the training set
            printDataSet(trainingData, "Training Dataset");
            printDataSet(testData, "Test Dataset");

            /*
            * Classify Test Instance...
            *
            * Expected output
            *
            * Instance 1: 1.0 (Raja)
            * Instance 2: 0.0 (Truncata)
            * Instance 3: 0.0 (Truncata)
            * Instance 4: 0.0 (Truncata)
            * */
            System.out.println("\nClassifying test instances");
            int i = 1;
            String predictedClass;
            for(Instance instance : testData) {
                predictedClass = knn.classifyInstance(instance) == 0.0 ? "Truncata" : "Raja";
                System.out.println("Instance " + i + ": " + knn.classifyInstance(instance) + " (" + predictedClass + ")");
                i++;
            }

            /*
            * Standardise the data
            *
            * Expected output:
            *
            *  1.445388282, -0.222597281, N. truncata
2           * -1.032420202,  0.445194562, N. raja
3           *  0.20648404,   0.445194562, N. truncata
4           * -0.412968081, -1.558180968, N. raja
5           * -0.412968081, -0.890389124, N. raja
6           * -1.032420202, -0.890389124, N. raja
7           *  0.825936161,  1.112986405, N. raja
8           * -0.412968081,  1.112986405, N. truncata
9           *  0.825936161, -0.222597281, N. raja
10          *  0.20648404,  -0.890389124, N. truncata
11          * -1.651872323, -0.222597281, N. raja
12          *  1.445388282,  1.780778249, N. truncata
            * */
            System.out.println("\nStandardising the training data");
            ((KNN) knn).setStandardise(true);
            knn.buildClassifier(trainingData);
            for(Instance si : ((KNN) knn).getTrainingData()){
                System.out.println(si);
            }

            /*
            * Reclassify the standardised data
            *
            * Expected output:
            *
            * Instance 1: 0.0 (Truncata)
            * Instance 2: 0.0 (Truncata)
            * Instance 3: 0.0 (Truncata)
            * Instance 4: 0.0 (Truncata)
            *
            * */
            System.out.println("\nClassifying test instances with standardised data");
            i = 1;
            for(Instance instance : testData) {
                predictedClass = knn.classifyInstance(instance) == 0.0 ? "Truncata" : "Raja";
                System.out.println("Instance " + i + ": " + knn.classifyInstance(instance) + " (" + predictedClass + ")");
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void printDataSet(Instances data, String title){
        System.out.println("\n" + title);
        int caseID = 1;
        for(Instance i : data) {
            System.out.println(caseID + ") " + i);
            caseID++;
        }
    }
}

/*
* print test data, print training data
* classify each instance
* Show classified and show expected
* show standardised data
* standardise
* classify again and show results and expected
* */