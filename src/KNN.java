/**
 * Author:  100086865
 *
 * Represents a KNN classidier that extends the WEKA AbstractClassifier
 * Offers functionality to predict a class value and calculates the distribution
 * for a given instance. Extra functionality includes altering the value of k,
 * standardising the training data, leave one out cross validation and weighted voting schemes.
 */

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class KNN extends AbstractClassifier {

    public Instances getTrainingData() {
        return trainingData;
    }

    public void setTrainingData(Instances trainingData) {
        this.trainingData = trainingData;
    }

    //classifier fields
    private Instances trainingData;
    private Instances standardisedTrainingData;
    private int k;
    private double[] classRepresentationTally;

    //config flags
    private boolean standardise;
    private boolean loocValidation;
    private boolean weightedVoting;

    /**
     * default constructor that sets the value of k to 1 and sets the
     * config flags to their default value
     */
    public KNN(){
        this.k = 1;
        this.standardise = true;
        this.loocValidation = false;
        this.weightedVoting = false;
    }

    /**
     * constructor that sets the value of k to a custom value
     * @param k - value of k
     * @param standardise - flag that enables/disables data standardisation
     * @param loocValidation - flag that enables/disables leave one out cross validation
     * @param weightedVoting = flag that enables/disables a weighted voting scheme
     */
    public KNN(int k, boolean standardise, boolean loocValidation, boolean weightedVoting) {
        this.k = k;
        this.standardise = standardise;
        this.loocValidation = loocValidation;
        this.weightedVoting = weightedVoting;
    }

    /**
     * stores the training data
     * @param instances - training data
     * @throws Exception - thrown if the data set file cannot be found
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances standardisedTrainingData = new Instances(instances);

        //set k with looc validation
        if(this.loocValidation){
            //set k to 20% of of training size or 100, whichever is smaller
            double maxK =Math.min(instances.numInstances()*(20-100), 100);
            //values between 1 and maxK are the validation set
            //remaining instances are the training set
            //do a thing to get the best value of k
            //increment k from 1-maxK and do some tests to find the best value of k
        }

        //standardises the training data
        if(this.standardise) {
            for(int i = 0; i < instances.numAttributes()-1; i++){

                double mean = ClassifierTools.getMean(instances, i);
                double stddev = ClassifierTools.getStandardDeviation(instances, i);

                //loop through each value and standardise it
                for(Instance in : standardisedTrainingData){
                    double standardisedValue = (in.value(i) - mean) / stddev;
                    in.setValue(i, standardisedValue);
                }
            }
            this.trainingData = standardisedTrainingData;
        }
        //Un-standardised
        else {
            this.trainingData = instances;
        }
    }

    public boolean isStandardise() {
        return standardise;
    }

    public void setStandardise(boolean standardise) {
        this.standardise = standardise;
    }

    public boolean isLoocValidation() {
        return loocValidation;
    }

    public void setLoocValidation(boolean loocValidation) {
        this.loocValidation = loocValidation;
    }

    public boolean isWeightedVoting() {
        return weightedVoting;
    }

    public void setWeightedVoting(boolean weightedVoting) {
        this.weightedVoting = weightedVoting;
    }

    /**
     * Predicts the category of a given instance using the k-NN algorithm
     * @param testInstance - instance to predict the category of
     * @return the predicted class valye
     */
    @Override
    public double classifyInstance(Instance testInstance) {

        //K closest neighbours
        ArrayList<Instance> kNearestNeighbours = new ArrayList<>();
        //A copy of the training dat
        Instances tempTrainingData = new Instances(this.trainingData);

        //Get the k nearest neighbours to the test care and store them in a list
        for(int i = 0; i < this.k; i++){

            Instance currentSmallest = tempTrainingData.firstInstance();
            int currentSmallestIndex = 0;

            for(int j = 0; j < tempTrainingData.size(); j++){

                double distance = ClassifierTools.getDistance(tempTrainingData.get(j), testInstance);

                if(ClassifierTools.getDistance(currentSmallest, testInstance) > distance){
                    currentSmallest = tempTrainingData.get(j);
                    currentSmallestIndex = j;
                }
                //if many distances are equally as close then choose one at random to be the closest
                else if (ClassifierTools.getDistance(currentSmallest, testInstance) == distance){
                    if(ThreadLocalRandom.current().nextInt(0, 1 + 1) > 0.5){
                        currentSmallest = tempTrainingData.get(j);
                        currentSmallestIndex = j;
                    }
                }
            }
            tempTrainingData.remove(currentSmallestIndex);
            kNearestNeighbours.add(currentSmallest);
        }

        double [] classRepresentationTally = new double[this.trainingData.numClasses()];

        for(Instance x : kNearestNeighbours){
        }

        //give each neighbour an equal vote
        for (Instance key : kNearestNeighbours) {
            classRepresentationTally[(int) key.classValue()]++;
        }

        this.classRepresentationTally = classRepresentationTally;
        return ClassifierTools.findHigestTally(classRepresentationTally);
    }

    /**
     * Returns the class proprtion of neighbours that voted for a particular class
     * @return class distribution as double array
     */
    //TODO: This should be proportion. i.e. (1/5%, 2/5%, 2/5%)
    public double[] distributionForInstance(){
        return this.classRepresentationTally;
    }

    /**
     * sets the k value
     * @param k - how many closest neighbours the classifier uses
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
}