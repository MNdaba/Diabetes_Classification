package code;

/**Enhanced K Means by excluding outliers by using the distance measure from the origin
//The algorithm searches for an optimal K within a given range*/

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Scanner;

public class TestKMeans {

	/**
	 * @param args
	 * @throws FileNotFoundException
	 * 
	 * 
	 */
	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;
	public static double countZero = 0;
	public static double countOne = 0;

	public static void setCentroidLabel(ArrayList<Cluster> cluster) {
		int countZero = 0;
		int countOne = 0;
		for (Cluster c : cluster) {
			countZero = 0;
			countOne = 0;
			for (Observation o : c.getObservations()) {
				if (o.getLabel() == 1)
					countOne++;
				else
					countZero++;
			}
			if (countOne > countZero)
				c.centroid.setLabel(1);
			else
				c.centroid.setLabel(0);
		}
	}

	public static double getAccuracy(ArrayList<Cluster> clusters,
			ArrayList<Observation> testSet, int numOfClusters, boolean test) {
		double max = Double.MAX_VALUE;
		double min = max;
		double distance = 0.0;
		int correct = 0;
		int cluster = 0;
		// TN = 0;
		// TP = 0;
		// FN = 0;
		// FP = 0;
		countZero = 0;
		countOne = 0;
		for (Observation obs : testSet) {
			min = max;
			for (int j = 0; j < numOfClusters; j++) {
				Cluster c = clusters.get(j);
				distance = Observation.distance(obs, c.getCentroid());
				if (distance < min) {
					min = distance;
					cluster = j;
				}
			}
			int clusterlabel = clusters.get(cluster).centroid.getLabel();

			if (clusterlabel == obs.getLabel()) {
				if (test == true) {
					if (clusterlabel == 0 && obs.getLabel() == 0)
						TN++;
					else
						TP++;
				}
				correct++;
			} else {
				if (test == true) {
					if (clusterlabel == 0 && obs.getLabel() == 1)
						FN++;
					else
						FP++;
				}
			}

		}
		return correct / (double) (testSet.size());
	}

	public static String getAccuracy(String datasetPath, int numOfFolds,int kValue) throws FileNotFoundException {
		DecimalFormat numberFormat = new DecimalFormat("#.000");
		long startTime = System.currentTimeMillis();
		double finalR = 0;
		double bestSolution = 0;
		String bestSolutionMetrics = "";
		ArrayList<Observation> list = KMeans.readDataSet(datasetPath);//Read in Dataset
		String result = "";
		Observation[][] fold = new Observation[numOfFolds][(int) (list.size() / numOfFolds)];
		TN = 0;
		TP = 0;
		FN = 0;
		FP = 0;
		int index = 0;
		double[] accuracies = new double[numOfFolds];
		int count = 0;
		double[] arr = new double[list.size()];
		int quartileInd = 0;
				for (Observation o : list) 
			        o.setD(Observation.distance(o,new Observation(new double[o.getVars().length], 0)));
					
				Collections.shuffle(list);
				for (int i = 0; i < numOfFolds; i++) {

					for (int j = 0; j < (int) (list.size() / numOfFolds); j++) {
						fold[i][j] = list.get(count);
						count++;
					}
				}
				ArrayList<Observation> trainSet = new ArrayList<Observation>();
				ArrayList<Observation> validationSet = new ArrayList<Observation>();
				ArrayList<Observation> testSet = new ArrayList<Observation>();
				ArrayList<ArrayList<Observation>> data = new ArrayList<ArrayList<Observation>>();

				 
				double accuracy = 0;
				ArrayList<Observation> finalSet = new ArrayList<Observation>();
				data.clear();
				finalSet.clear();
				trainSet.clear();
				testSet.clear();
				for (int i = 0; i < numOfFolds; i++) {
					data.add(new ArrayList<Observation>());
				}
				for (int i = 0; i < numOfFolds; i++) {
					for (int j = 0; j < (int) (list.size() / numOfFolds); j++) {
						data.get(i).add(fold[i][j]);
					}
				}

				for (int i = 0; i < numOfFolds; i++) { // k-fold cv and classification 		 
					validationSet = data.get(0);
					data.remove(0);

					for (int j = 0; j < numOfFolds - 1; j++) {
						for (int k = 0; k < (int) (list.size() / numOfFolds); k++) {
							trainSet.add(data.get(j).get(k));// add training set
																// from the
																// remaining
																// samples
						}
					}// add all for learning cluster centres

					KMeans kmeans = new KMeans(kValue,
							trainSet.size(), trainSet);
					kmeans.init(kValue);
					boolean calculated = kmeans.calculate();
					ArrayList<Cluster> clusters = kmeans.getClusters();
					setCentroidLabel(clusters);

					accuracies[index] = getAccuracy(clusters, validationSet, kValue,true); // Accuracy on Validation Set
					data.add(data.size() - 1, validationSet);
					trainSet = new ArrayList<Observation>();
					validationSet = new ArrayList<Observation>();
					index++;

				}
				double sumT = 0;
				for (int a = 0; a < numOfFolds; a++) sumT += accuracies[a]; //To compute AVG k-fold cv accuracy
	
				finalR = ((double) sumT / accuracies.length);
				result = "";
				long endTime = System.currentTimeMillis();
				double totalTime = ((endTime - startTime) / 1000.0000);

				result += "K-Means Classification Accuracy: "
						+ numberFormat
								.format((100 * (sumT / (accuracies.length * 1.0))))
						+ "% \n \n";
				result += "Sensitivity: "
						+ Math.round(100.0 * ((+(TP) / ((double) (TP + FN)))))
						+ " % \n \n";
				;
				result += ("Specificity: " + Math
						.round(100.0 * ((TN) / (double) (TN + FP))))
						+ " % \n \n";
				result += ("Positive Predictive Value: " + Math
						.round(100.0 * ((TP) / (double) (TP + FP))))
						+ " % \n \n";
				if (finalR > bestSolution) {
					bestSolution = finalR;
					bestSolutionMetrics = "";
					bestSolutionMetrics += "Sensitivity: "
							+ Math.round(100.0 * ((+(TP) / ((double) (TP + FN)))))
							+ " % \n \n";
					;
					bestSolutionMetrics += ("Specificity: " + Math
							.round(100.0 * ((TN) / (double) (TN + FP))))
							+ " % \n \n";
					bestSolutionMetrics += ("Positive Predictive Value: " + Math
							.round(100.0 * ((TP) / (double) (TP + FP))))
							+ " % \n \n";
					bestSolutionMetrics += ("Negative Predictive Value: " + Math
							.round(100.0 * ((TN) / (double) (FN + TN))))
							+ " % \n \n";
					}
				if (finalR >= 0.78) {
					result += "\n";
					return result;

				}// clusters
		endTime = System.currentTimeMillis();
		totalTime = ((endTime - startTime) / 1000.0000);
		String finalResult = "K-Means Classification Accuracy: "
				+ numberFormat.format(100.0 * (bestSolution)) + "% \n \n";
		finalResult += bestSolutionMetrics;
		finalResult += "\n";
		return finalResult;

	}

	public static double quartile(double[] values, double lowerPercent) {

		if (values == null || values.length == 0) {
			throw new IllegalArgumentException(
					"The data array either is null or does not contain any data.");
		}

		// Rank order the values
		double[] v = new double[values.length];
		System.arraycopy(values, 0, v, 0, values.length);
		Arrays.sort(v);

		int n = (int) Math.round(v.length * lowerPercent / 100);

		return (v[n]);

	}

	public static void main(String[] args) throws IOException {

		String dataset ="";
		String result ="";
	    dataset = "B-[Excl Missing].txt";
		//dataset = "D-[Extracted Features].txt";
		//dataset = "C-[Replaced by Mean].txt";
		//dataset = "A-[Unprocessed].txt";
	
	    for(int i=0;i<15;i++) {
	    	System.out.println("-------------- Experiment: "+(i+1)+" --------------");
		System.out.println(getAccuracy(dataset,9,10));
	    System.out.println();
	    }
		
		
	}

}
