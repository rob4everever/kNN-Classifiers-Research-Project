import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class KNN extends AbstractClassifier {

    private Instances trainingData;
    private int k;

    /**
     * default constructor that sets the value of k to 1
     */
    public KNN(){
        this.k = 1;
    }

    /**
     * constructor that sets the value of k to a custom value
     * @param k - value of k
     */
    public KNN(int k) {
        this.k = k;
    }

    /**
     * stores the training data
     * @param instances - training data
     * @throws Exception - thrown if the data set file cannot be found
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.trainingData = instances;
    }

    /**
     * Predicts the category of a given instance using the k-NN algorithm
     * @param instance - instance to predict the category of
     * @return the predicted class valye
     */
    @Override
    public double classifyInstance(Instance instance) {

        HashMap<Instance, Double> kClosestInstances = new HashMap<>();

        //loop through the training data
        //calculate the distance between all training instance and the test instance
        //find the smallest
        //add the smallest to the map along with its distance
        //repaeat k times
        for

        print(map);
        return 2;
    }

    /**
     * Calculates the Euclidean distance between two instances
     * @param x - Instance 1
     * @param y - Instance 2
     * @return Euclidean distance
     */
    private double getDistance(Instance x, Instance y) {

        double distance = 0;

        for(int i = 0; i < x.numAttributes()-1; i++){
            distance += Math.pow(x.value(i) - y.value(i), 2);
        }

        return Math.sqrt(distance);
    }

    /**
     * sets the k value
     * @param k
     */
    public void setK(int k) {
        this.k = k;
    }

    /**
     * return the k value
     * @return k
     */
    public int getK() {
        return this.k;
    }




    public static void print(Map<Instance, Double> map) {
        if (map.isEmpty())  {
            System.out.println("map is empty");
        }
        else {
            System.out.println(map);
        }
    }
}