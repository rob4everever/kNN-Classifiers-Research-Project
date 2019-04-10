import java.io.FileReader;
import weka.core.Instances;

public class ClassifierTools {

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
}
