/**
 * Author:  100086865
 *
 * Represents a KNN classidier that extends the WEKA AbstractClassifier
 * Offers functionality to predict a class value and calculates the distribution
 * for a given instance. Extra functionality includes altering the value of k,
 * standardising the training data, leave one out cross validation and weighted voting schemes.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class KNN extends AbstractClassifier {

    //Classifier settings
    private int k;
    private Instances trainingData;
    private Instances standardisedTrainingData;
    private double[] distributionForInstance;

    //Classifier config flags
    private boolean standardised;
    private boolean LOOCValidation;
    private boolean weightedVoting;

    /**
     * Creates a KNN classifier with default values
     */
    public KNN(){
        //default classifier settings
        this.k = 1;
        this.standardised = false;
        this.LOOCValidation = false;
        this.weightedVoting = false;
    }

    /**
     * Trains the classifier with the training data or a standardised
     * version of the training data
     * @param instances - training data
     * @throws Exception NullPointerException
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {

        if(this.standardised){

            this.standardisedTrainingData = new Instances(this.trainingData);

            for(int i = 0; i < instances.numAttributes()-1; i++){

                double mean = ClassifierTools.getMean(instances, i);
                double stddev = ClassifierTools.getStandardDeviation(instances, i);

                //loop through each value and standardise it
                for(Instance trainingInstance: this.standardisedTrainingData){
                    double standardisedValue = (trainingInstance.value(i) - mean) / stddev;
                    trainingInstance.setValue(i, standardisedValue);
                }
            }
        }
        else{
            this.trainingData = instances;
        }
    }

    /**
     *
     * @param testInstance
     * @return
     */
    @Override
    public double classifyInstance(Instance testInstance){

        ArrayList<Instance> kNearestNeighbours = getKNearestNeighbours(testInstance);

        //determine most represented class in list
        double[] classTally = new double[this.trainingData.numClasses()];
        Arrays.fill(classTally, 0.0);
        for(Instance instance : kNearestNeighbours){
            if(this.weightedVoting){
                double voteWeight = 1/(ClassifierTools.getDistance(instance, testInstance));
                classTally[(int)instance.classValue()]+=voteWeight;
            }
            else{
                classTally[(int)instance.classValue()]++;
            }
        }

        this.distributionForInstance = classTally;
        return ClassifierTools.findHigestTally(classTally);
    }

    /**
     * Calculate the proportion of class representation
     * @param instance
     * @return
     */
    @Override
    public double[] distributionForInstance(Instance instance){

        double[] classTally = new double[this.trainingData.numClasses()];

        for(int i = 0; i < distributionForInstance.length; i++){
            classTally[i] = distributionForInstance[i] / this.k;
        }

        return classTally;
    }

    /**
     * Finds the k nearest neighbours to an instance
     * @param instance
     * @return k closest neighbours
     */
    public ArrayList<Instance> getKNearestNeighbours(Instance instance){

        ArrayList<Instance> kNearestNeighbours = new ArrayList<>();
        Instances tempTrainingData;

        if(this.standardised){
            tempTrainingData = new Instances(this.standardisedTrainingData);
        }
        else{
            tempTrainingData = new Instances(this.trainingData);

        }

        //find K nearest neighbours
        for(int i = 0; i < this.k; i++){

            Instance currentSmallest = tempTrainingData.firstInstance();
            int currentSmallestIndex = 0;

            for(int j = 0; j < tempTrainingData.size(); j++) {
                double distance = ClassifierTools.getDistance(tempTrainingData.get(j), instance);
                if (ClassifierTools.getDistance(currentSmallest, instance) > distance) {
                    currentSmallest = tempTrainingData.get(j);
                    currentSmallestIndex = j;
                }
                //if multiple neighbours have the same distance then pick one at random
                else if (ClassifierTools.getDistance(currentSmallest, instance) == distance){
                    if(ThreadLocalRandom.current().nextInt(0, 1 + 1) > 0.5){
                        currentSmallest = tempTrainingData.get(j);
                        currentSmallestIndex = j;
                    }
                }
            }
            kNearestNeighbours.add(currentSmallest);
            tempTrainingData.delete(currentSmallestIndex);
        }

        return kNearestNeighbours;
    }

    /**
     * Get the value of k
     * @return
     */
    public int getK() {
        return k;
    }

    /**
     * Set the value of k
     * @param k
     */
    public void setK(int k) {
        this.k = k;
    }

    /**
     * Get training data
     * @return training data
     */
    public Instances getTrainingData() {
        return trainingData;
    }

    /**
     * Get standardised training data
     * @return standardised data
     */
    public Instances getStandardisedTrainingData() {
        return standardisedTrainingData;
    }

    /**
     *
     * @param standardisedTrainingData
     */
    public void setStandardisedTrainingData(Instances standardisedTrainingData) {
        this.standardisedTrainingData = standardisedTrainingData;
    }

    /**
     *
     * @return
     */
    public double[] getDistributionForInstance() {
        return distributionForInstance;
    }

    /**
     *
     * @param distributionForInstance
     */
    public void setDistributionForInstance(double[] distributionForInstance) {
        this.distributionForInstance = distributionForInstance;
    }

    /**
     * Return true if standardising is set, otherwise false
     * @return
     */
    public boolean isStandardised() {
        return standardised;
    }

    /**
     *
     * @param standardised
     */
    public void setStandardised(boolean standardised) {
        this.standardised = standardised;
    }

    /**
     * Return true if LOOC validation is set, otherwise false
     * @return
     */
    public boolean isLOOCValidation() {
        return LOOCValidation;
    }

    /**
     *
     * @param LOOCValidation
     */
    public void setLOOCValidation(boolean LOOCValidation) {
        this.LOOCValidation = LOOCValidation;
    }

    /**
     * Return true if weighted voting is set, otherwise false
     * @return if weighted voting
     */
    public boolean isWeightedVoting() {
        return weightedVoting;
    }

    /**
     * Set weighted voting
     * @param weightedVoting
     */
    public void setWeightedVoting(boolean weightedVoting) {
        this.weightedVoting = weightedVoting;
    }

    /**
     * Returns the classifier as a string
     * @return clasifer configuration
     */
    @Override
    public String toString() {
        return "KNN{" +
            "k=" + k +
            ", standardised=" + standardised +
            ", LOOCValidation=" + LOOCValidation +
            ", weightedVoting=" + weightedVoting +
            '}';
    }
}