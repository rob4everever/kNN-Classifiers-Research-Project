/**
 * Author:  100086865
 *
 * A set of static methods that provide useful functionality to the
 * classifier.
 */

import java.io.FileReader;
import java.util.concurrent.ThreadLocalRandom;
import org.jetbrains.annotations.Nullable;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierTools {

    /**
     * Loads an .arff file of data into memory
     * @param filePath - location of .arff file
     * @return Instances object
     */

    @Nullable
    public static Instances loadClassifierData(String filePath) {

        Instances data;

        try {
            FileReader reader = new FileReader(filePath);
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (Exception e) {
            System.out.println("Exception Caught: " + e);
        }

        return null;
    }

    /**
     * Calculates the Euclidean distance between two instances
     * @param x - Instance 1
     * @param y - Instance 2
     * @return Euclidean distance
     */
    public static double getDistance(Instance x, Instance y) {

        double distance = 0;

        for(int i = 0; i < x.numAttributes()-1; i++){
            distance += Math.pow(x.value(i) - y.value(i), 2);
        }

        return Math.sqrt(distance);
    }

    /**
     * Finds the most tallied value in a tally array
     * @param classRepresentationTally
     * @return the index with the highest tally
     */
    public static double findHigestTally(double[] classRepresentationTally){

        int highestTallyIndex = 0;
        double highestValue = classRepresentationTally[0];
        //Loop through the tally and find the most represented class
        for(int i = 1; i < classRepresentationTally.length; i++){
            if(classRepresentationTally[i] > classRepresentationTally[highestTallyIndex]){
                highestTallyIndex = i;
            }
        }
        return highestTallyIndex;
    }

    /**
     * Returns the mean of a column of data
     * @param data - data to calculate mean of
     * @param attributeNum - attribute column index to calculate the mean of
     * @return the mean of a set of values
     */
    public static double getMean(Instances data, int attributeNum){

        double sum = 0.0;
        int numOfInstances = data.numInstances();
        for(int i = 0; i < data.size(); i++){
            sum += data.instance(i).value(attributeNum);
        }
        return sum/numOfInstances;
    }

    /**
     * Returns the standard deviation of a column of data
     * @param data - data to calculate standard deviation of
     * @param attributeNum - attribute column index to calculate the mean of
     * @return the standard deviation of a set of values
     */
    public static double getStandardDeviation(Instances data, int attributeNum) {

        double stdDev = 0.0;
        double mean = getMean(data, attributeNum);
        int numOfInstances = data.numInstances();

        for(Instance trainingInstance : data){
            stdDev += Math.pow(trainingInstance.value(attributeNum) - mean, 2);
        }

        stdDev = stdDev / (numOfInstances-1);
        return Math.sqrt(stdDev);
    }
}