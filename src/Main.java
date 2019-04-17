/**
 * Author:  100086865
 *
 * Entry point for the classifier. Data is provided to the classifier here
 * as well as showcasing the results. The class has two test harnests:
 */

import weka.core.Instance;
import weka.core.Instances;

public class Main {

    public static void main(String[]args) throws Exception {

        final String ROOT = "res/";
        final String TRAINING_DATA = "Pitcher_Training.arff";
        final String TEST_DATA = "Pitcher_Test.arff";

        //Load the training data
        Instances trainingData = ClassifierTools.loadClassifierData(ROOT + TRAINING_DATA);
        Instances testData = ClassifierTools.loadClassifierData(ROOT + TEST_DATA);

        KNN KNNClassifier = new KNN();
        KNNClassifier.buildClassifier(trainingData);

        /**
         * Classify the test data using the training data provided
         * in part 1 of the coursework
         *
         * Expected output
         *
         * Instance 1: 1.0(Raja)
         * Instance 2: 0.0(Truncata)
         * Instance 3: 0.0(Truncata)
         * Instance 4: 0.0(Truncata)
         */
        System.out.println("\nClassifying test data...");
        int i = 1;
        for(Instance testInstance : testData){
            String className;
            double prediction = KNNClassifier.classifyInstance(testInstance);
            if(prediction == 0.0){ className = "Truncata"; }
            else{ className = "Raja"; }
            System.out.println("Instance " + i + ": " + prediction + "(" + className + ")");
            i++;
        }

        /**
         * Classify the test data using a standardised version of
         * the data provided in part 1 of the coursework
         *
         * Expected output
         *
         * Instance 1: 0.0(Truncata)
         * Instance 2: 0.0(Truncata)
         * Instance 3: 0.0(Truncata)
         * Instance 4: 0.0(Truncata)
         */
        System.out.println("\nClassifying standardised test data...");
        KNNClassifier.setStandardised(true);
        KNNClassifier.buildClassifier(trainingData);
        i = 1;
        for(Instance testInstance : testData){
            String className;
            double prediction = KNNClassifier.classifyInstance(testInstance);
            if(prediction == 0.0){ className = "Truncata"; }
            else{ className = "Raja"; }
            System.out.println("Instance " + i + ": " + prediction + "(" + className + ")");
            i++;
        }

        /**
         * Classify the test data using a standardised version of the data
         * provided in part 1 of the coursework using weighted voting
         *
         * Expected output
         *
         * Instance 1: 1.0(Raja)
         * Instance 2: 0.0(Truncata)
         * Instance 3: 0.0(Truncata)
         * Instance 4: 1.0(Raja)
         */
        System.out.println("\nClassifying test data with weighted voting...");
        KNNClassifier.setStandardised(false);
        KNNClassifier.setWeightedVoting(true);
        KNNClassifier.buildClassifier(trainingData);
        i = 1;
        for(Instance testInstance : testData){
            String className;
            double prediction = KNNClassifier.classifyInstance(testInstance);
            if(prediction == 0.0){ className = "Truncata"; }
            else{ className = "Raja"; }
            System.out.println("Instance " + i + ": " + prediction + "(" + className + ")");
            i++;
        }

        /**
         * Classify the test data using a standardised version of the data
         * provided in part 1 of the coursework using a KNN ensemble
         *
         * Expected Output
         *
         * Instance 1: 1.0(Raja)
         * Instance 2: 0.0(Truncata)
         * Instance 3: 0.0(Truncata)
         * Instance 4: 1.0(Raja)
         */
        System.out.println("\nKNN Ensemble...");
        KNNEnsemble KNNensemble = new KNNEnsemble();
        KNNensemble.buildClassifier(trainingData);
        i = 1;
        for(Instance testInstance : testData){
            String className;
            double prediction = KNNensemble.classifyInstance(testInstance);
            if(prediction == 0.0){ className = "Truncata"; }
            else{ className = "Raja"; }
            System.out.println("Instance " + i + ": " + prediction + "(" + className + ")");
            i++;
        }
    }
}