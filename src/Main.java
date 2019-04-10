import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Main {

    public static void main(String[]args){

        Instances trainingData = ClassifierTools.loadClassifierData("res/FootballPlayers.arff");
        Instances testData = ClassifierTools.loadClassifierData("res/FootballPlayers_TEST.arff");
        Classifier knn = new KNN(3);
        Instance i = testData.firstInstance();

        try {
            knn.buildClassifier(trainingData);
            System.out.println(knn.classifyInstance(i));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
