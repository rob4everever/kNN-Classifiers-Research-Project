/**
 * Author:  100086865
 *
 * Class description here
 */

import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;

public class KNNEnsemble {

    private ArrayList<KNN> classifiers;
    private int ensembleSize;
    private Instances trainingData;

    /**
     * Creates a KNN ensemble with the default configuration
     */
    public KNNEnsemble(){
        //Ensemble default settings
        this.ensembleSize = 50;
        this.classifiers = new ArrayList<KNN>();
    }

    /**
     * Builds an emsemble of N KNN classifiers
     * @param instances - Training data
     */
    public void buildClassifier(Instances instances)  {

        //store the training data
        this.trainingData = instances;

        //Train each classifier with a random sample of the training data
        for(int i = 0; i < this.ensembleSize; i++){
            Instances trainingSubset = new Instances(trainingData);
            for(int j = 0; j < this.trainingData.size(); j++){
                Random r = new Random();
                int randomInt = r.nextInt(this.trainingData.size());
                trainingSubset.set(j ,this.trainingData.instance(randomInt));
            }
            KNN KNNClassifier = new KNN();
            try {
                KNNClassifier.buildClassifier(trainingSubset);
            } catch (Exception e) {
                e.printStackTrace();
            }
            this.classifiers.add(KNNClassifier);
        }
    }

    /**
     * Classifies a test instance with each classifier in the
     * ensemble and combines the output to get a final output
     * @param testInstance - Instance to classify
     * @return the final prediction
     */
    public double classifyInstance(Instance testInstance){

        double[] classTally = new double[this.trainingData.numClasses()];
        for(int i = 0; i < this.ensembleSize; i++){
            double prediction = this.classifiers.get(i).classifyInstance(testInstance);
            classTally[(int)prediction]++;
        }
        return ClassifierTools.findHigestTally(classTally);
    }

    /**
     * Get the ensemble size
     * @return ensemble size
     */
    public int getEnsembleSize() {
        return ensembleSize;
    }

    /**
     * Set the ensemble size
     * @param ensembleSize - number of classifiers in the ensemble
     */
    public void setEnsembleSize(int ensembleSize) {
        this.ensembleSize = ensembleSize;
    }

    /**
     * Get the training data
     * @return training data
     */
    public Instances getTrainingData() {
        return trainingData;
    }
}