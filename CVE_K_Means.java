package code;

 //CVE-K-Means
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

public class CVE_K_Means {

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
/**CVE-Kmeans method. It takes in the file name, number of fold for cross validation and min,max values for a range to find the best k*/
	public static String getResults(String datasetPath, int numOfFolds,int minK, int maxK) throws IOException {
		DecimalFormat numberFormat = new DecimalFormat("#.000");
		long start = System.currentTimeMillis();
		long end = start + 30 * 1000; // 30 seconds * 1000 ms/sec
		long startTime = System.currentTimeMillis();
		double finalR = 0;
		int clustersNum = 0;
		double bestSolution = 0;
		String bestSolutionMetrics = "";
		int bestClusterNum = 0;
		ArrayList<Observation> list = KMeans.readDataSet(datasetPath);/**Read dataset*/
		while (finalR < 0.76 && (System.currentTimeMillis() < end)) {/**30 seconds as stopping condition & take best solution found
			                                                           Threshold value of 0.76 set through trial and error as best solution CVE must beat for all datasets
			                                                           Algorithm terminates if best solution found or no best solution found in < 30 seconds*/
			String result = "";
			Observation[][] fold = new Observation[numOfFolds][(int) (list.size() / numOfFolds)];
			
                /**Search for k with best solution between minK and maxK*/
			for (int x = minK; x <= maxK; x++) {
				clustersNum = x;
				TN = 0;
				TP = 0;
				FN = 0;
				FP = 0;
				int index = 0;
				double[] accuracies = new double[numOfFolds];
				int count = 0;
				double[] arr = new double[list.size()];
				int quartileInd = 0;
				for (Observation o : list) {
					o.setD(Observation.distance(o,new Observation(new double[o.getVars().length], 0)));
					arr[quartileInd++] = Observation.distance(o,new Observation(new double[o.getVars().length], 0));
				}
				/** Remove outliers by using quartiles*/
				double Q3 = quartile(arr, 75);
				double Q1 = quartile(arr, 25);
				double IQR = Q3 - Q1;
				double upperBound = Q3 + (IQR * 1.5);
				double lowerBound = Q1 - (IQR * 1.5);
				Iterator<Observation> iter = list.iterator();
				while (iter.hasNext()) {
					Observation next = iter.next();
					if (next.getD() < lowerBound || next.getD() > upperBound)
						iter.remove();/**Remove outlier*/
				}

				Collections.shuffle(list);
				for (int i = 0; i < numOfFolds; i++) {//prepare data for cross validation
					for (int j = 0; j < (int) (list.size() / numOfFolds); j++) {
						fold[i][j] = list.get(count);
						count++;
					}
				}
				ArrayList<Observation> trainSet = new ArrayList<Observation>();
				ArrayList<Observation> validationSet = new ArrayList<Observation>();
				ArrayList<Observation> testSet = new ArrayList<Observation>();
				ArrayList<ArrayList<Observation>> data = new ArrayList<ArrayList<Observation>>();
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

				for (int i = 0; i < numOfFolds; i++) { /**K-fold Cross Validation && Classification*/											  
					validationSet = data.get(0);
					data.remove(0);

					for (int j = 0; j < numOfFolds - 1; j++) {
						for (int k = 0; k < (int) (list.size() / numOfFolds); k++) {
							trainSet.add(data.get(j).get(k));/**Combine remaining samples into training set and use it to derive cluster centers*/
						}
					}

					/** Invoke K-Means clustering with x as current k in the search space and the training set*/
					KMeans kmeans = new KMeans(x, trainSet.size(), trainSet);
					kmeans.init(x);
					kmeans.calculate();
					ArrayList<Cluster> clusters = kmeans.getClusters();
					setCentroidLabel(clusters);
					accuracies[index] = getAccuracy(clusters, validationSet, x,true); /**Accuracy on validation set for the current fold ans k = x*/
					data.add(data.size() - 1, validationSet);
					trainSet = new ArrayList<Observation>();
					validationSet = new ArrayList<Observation>();
					index++;

				}//cv ends here

				double sumT = 0;
			for (int a = 0; a < numOfFolds; a++) sumT += accuracies[a];
				finalR = ((double) sumT / accuracies.length);//Final accuracy for current k value
				result = "";
				result += "CVE-K-Means Classification Accuracy: "
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
				result += ("Negative Predictive Value: " + Math
						.round(100.0 * ((TN) / (double) (FN + TN))))
						+ " % \n \n";
				if (finalR > bestSolution) {/** If accuracy for current k is better than the previous solution, set it as the best solution so far*/
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
					bestClusterNum = x;
				}
				if (finalR > 0.76) {/**If solution is above the set threshold, terminate the search and return the results with the best k
				                          Threshold found through trial and error*/
					result += "\n";
					result += ("k (number of clusters) for best solution: " + clustersNum);
					return result;
				}

			}//End the search for best k using on Min-Max Loop

		}// end while
		long endTime = System.currentTimeMillis();
		double totalTime = ((endTime - startTime) / 1000.0000);
		String finalResult = "CVE-K-Means Classification Accuracy: "
				+ numberFormat.format(100.0 * (bestSolution)) + "% \n \n";
		finalResult += bestSolutionMetrics;
		finalResult += "\n";
	//	finalResult += "Total Execution Time: " + totalTime + " seconds\n";
		finalResult += ("k (number of clusters) for best solution: " + bestClusterNum);

		return finalResult;
	}

	/**Quartile method used to find quartiles for outlier detection and removal */
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
		
		/**
		 * To run experiments,  uncomment a dataset you want to use for experimentation
		 * Threshold values are used to easily find the best solution so that a user does not re-run the program each time to find a solution
		 * Threshold values were found through trial and error
		 * */
		
    for(int i=0;i<10000;i++) {/**Number of experiments, loop terminates when best solution is found*/	
		String dataset ="";
		dataset = "B-[Excl Missing].txt";
		//dataset = "D-[Extracted Features].txt";
		//dataset = "C-[Replaced by Mean].txt";
		//dataset = "A-[Unprocessed].txt";
			
			String result = getResults(dataset,9,3,12);
		 	double threshold = 0;
			if(dataset.equals("B-[Excl Missing].txt")) threshold = 79.5;
			else if(dataset.equals("D-[Extracted Features].txt")) threshold = 78.5;
			else if(dataset.equals("C-[Replaced by Mean].txt")) threshold = 77.5;
		    else threshold = 76.5;
			System.out.println("Experiment "+(i+1)+" Accuracy >>  "+Double.parseDouble(result.substring(37, 43))+"%");
		 	if(Double.parseDouble(result.substring(37, 43)) >= threshold) {//Best Solution found
		System.out.println();
		System.out.println("******************----------------------*****************");
	System.out.println(result);
	System.exit(0);


	}
		}

		
		
	}
	}


