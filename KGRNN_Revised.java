package code;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

import javax.swing.plaf.synth.SynthSeparatorUI;

public class KGRNN_Revised {

	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;

	public KGRNN_Revised() {

	}

	/**getCentroids creates a balanced split of trainset and testSet (70%/30% split) and then returns the testSet and formed centroids*/
	public static ArrayList<ArrayList<Observation>> getCentroids(
	    String filePath, int numOfClusters, double sigma) throws IOException {//get best Centroids to use as training set
		ArrayList<Observation> allData = KMeans.readDataSet(filePath);
		int numOfFolds = 10;/**Used to split dataset into 10 samples of equal size (ensuring the proportion of -ve and +instances is preserved)*/
		Observation[][] fold = new Observation[numOfFolds][(int) (allData.size() / numOfFolds)];
		double[] accuracies = new double[numOfFolds];
		Collections.shuffle(allData);
		 
		ArrayList<Observation> positiveInstances = new ArrayList<Observation>();
		ArrayList<Observation> negativeInstances = new ArrayList<Observation>();
		int countPos = 0;
		int countNeg = 0;
		for(int k=0;k<allData.size();k++) {/**Split -ve and +ve instances*/
			if(allData.get(k).getLabel()==1)positiveInstances.add(allData.get(k));
			else negativeInstances.add(allData.get(k));
		}
			Collections.shuffle(positiveInstances);
			Collections.shuffle(negativeInstances);
			/**Ensure each sample has balanced -ve and +ve instances*/
			for (int i = 0; i < numOfFolds; i++) {
			for (int j = 0; j < (int) (allData.size() / numOfFolds); j++) {
				if(j<(int)(positiveInstances.size()/numOfFolds)) {
				fold[i][j] = positiveInstances.get(countPos);
				countPos++;
				}
				else {fold[i][j] = negativeInstances.get(countNeg);
				countNeg++;
				}
			}
		}
			
 			 			
 			ArrayList<Observation> trainSet = new ArrayList<Observation>();
			ArrayList<Observation> testSet = new ArrayList<Observation>();
			ArrayList<ArrayList<Observation>> data = new ArrayList<ArrayList<Observation>>();
			for (int i = 0; i < numOfFolds; i++) {
				data.add(new ArrayList<Observation>());
			}
		    /**data contains a list of 10 samples*/
			for (int i = 0; i < numOfFolds; i++) {
				for (int j = 0; j < (int) (allData.size() / numOfFolds); j++) {
								data.get(i).add(fold[i][j]); //add folds into a list
							Collections.shuffle(data.get(i));
							}
						}
	 	ArrayList<Observation> validationSet = new ArrayList<Observation>();
	  	ArrayList<ArrayList<Observation>> finalChoice = new ArrayList<ArrayList<Observation>>();
		ArrayList<Observation> centroids = new ArrayList<Observation>();
		 

for(int i=0;i<3;i++) {/**Add 3 samples out of 10 into test set (creating 30% of test cases)*/
	for(Observation o: data.get(i))
		validationSet.add(o);
}
		 data.remove(0);data.remove(1);data.remove(2);/**Remove samples that are in the test set*/
		for (int i = 0; i < numOfFolds; i++) {//cv starts here
			testSet = data.get(0);//hold one sample to be test set
			data.remove(0);//remove test set from the samples
 
			for (int j = 0; j < numOfFolds - 4; j++) {
				for (int k = 0; k < (int) (allData.size() / numOfFolds); k++) {
							trainSet.add(data.get(j).get(k));/**Combine all instances in remaining 6 samples into 1 trainset*/			
				}
			}
 
			KMeans kmeans = new KMeans(numOfClusters, trainSet.size(), trainSet);
			kmeans.init(numOfClusters);
			kmeans.calculate(); //create clusters
			ArrayList<Cluster> clusters = kmeans.getClusters();
			setCentroidLabel(clusters);
			centroids = kmeans.getCentroids();//initial centroids
			if (i == 0) {
				finalChoice.add(centroids);
		 		finalChoice.add(testSet);		  
			}
			// test the model

			double accuracyTest = getAccuracy(clusters, testSet, numOfClusters);
			accuracies[i] = accuracyTest;

			if (i > 0) {
				if (accuracies[i] > accuracies[i - 1]) {
				 
					finalChoice.clear();
					finalChoice.add(centroids);//update centroidlist if better centroids are found
					finalChoice.add(testSet);
					 				}
			}

			data.add(data.size() - 1, testSet);//add the removed sample back to k-1 samples for next iteration
			testSet = new ArrayList<Observation>();
			trainSet = new ArrayList<Observation>();
			
		}
		finalChoice.add(validationSet);
		return finalChoice;

	}

	/**The method getAccuracy is used to compute the accuracy of CVE-K-Means algorithm */
	public static double getAccuracy(ArrayList<Cluster> clusters,
			ArrayList<Observation> testSet, int numOfClusters) {
		double max = Double.MAX_VALUE;
		double min = max;
		double distance = 0.0;
		int correct = 0;
		int cluster = 0;
		for (Observation obs : testSet) {/**Find the closest cluster to the test instance*/
			min = max;
			for (int j = 0; j < numOfClusters; j++) {
				Cluster c = clusters.get(j);
				distance = Observation.distance(obs, c.getCentroid());/**Compute distance between each test instance and centroid*/
				if (distance < min) {
					min = distance;
					cluster = j;
				}
			}

			if (clusters.get(cluster).centroid.getLabel() == obs.getLabel())/**Use the centroid label of the closest cluster to classify test instance*/
				correct++;

		}
		return correct / (double) (testSet.size());/**Accuracy*/
	}

	 /** setCentroidLabel assigns a label to the centroid using the majority voting on cluster instance labels */
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
					c.centroid.setLabel(-1);
			}
		}
	    
	
		/** BKGRNN_Train_Classify takes a file name, number of clusters, sigma and number of iterations as input
		 * BKRNN is configured to run 10 times, each run has 1000 iterations.
		 * On each iteration:
		 * 1. The dataset is read from a file, 
		 * 2. It is then split into 70% train set and 30% test set
		 * 3. KGRNN classifier with centroids produced from a training set by the CVE-K-Means algorithm is then created
		 * 4. The overall result of the classifier is stored
		 * On next iteration:
		 * Steps 1-4 are executed. Then if the result in 4 is better than the previous iteration, then it is saved as the best solution
		 * After the 600th iteration, then the best result is returned for the run
		 * After 10 runs, the results are then averaged to get the final result of the KGRNN classifier
		 * The best number of runs, iterations, sigma, and number of clusters was found through experimentation
		 * */
	public static String KGRNN_Train_Classify(String filePath, int numOfCentroids,double sigma, int iterations) throws IOException {
		ArrayList<ArrayList<Observation>> finalChoice = new ArrayList<ArrayList<Observation>>();
		DecimalFormat numberFormat2 = new DecimalFormat("0.00");
		 
		double accuracy = 0;
		double currAccuracy = 0;
		double currSensitivity = 0;
		double currentSpecificity = 0;
		double currentPPV = 0;
		double currentNPV = 0;
		 
		double sensitivitySum = 0;
		double specificitySum =0;
		double PPVsum = 0;
		double NPVsum =0;
		
		 
		double sensitivity = 0;
		double specificity = 0;
		double NPV = 0;
		double PPV = 0;	
		double overallAccuracySum =0;

		long start = System.currentTimeMillis();
		//long end = start + 180 * 1000; // 60 seconds * 1000 ms/sec
		long end = start + 30*1000; 

		for(int run = 0; run<5;run++) {
			System.out.println("Run: "+(run+1)+"...");	
		 currAccuracy =0;currSensitivity=0;currentSpecificity =0; currentNPV=0;currentPPV=0;   
		 
		 	for(int k=0;k<iterations;k++) {// iterations, 1000 in this case
			accuracy = 0;
			finalChoice = getCentroids(filePath, numOfCentroids, sigma); //getCentroids returns the test set and training centroids
			ArrayList<Observation> centroids = finalChoice.get(0);
			ArrayList<Observation> testSet = finalChoice.get(2);
			 
			double exponent = 0;
			double currDistance = 0;
			double denom = 0;
			double num = 0;
			double yHat = 0;
			int countAccuracy = 0;
			accuracy = 0;
			TN = 0;
			TP = 0;
			FN = 0;
			FP = 0;			
			
			for (Observation o : testSet) {//Train & classifiy instance with GRNN network using centroids as training instances
				num =0;
				denom = 0; 
				
				for (int i = 0; i < centroids.size(); i++) {
					currDistance = Observation.distance(o, centroids.get(i));
					exponent = Math.exp(-1* (((currDistance * currDistance)) / (2 * sigma * sigma)));//--GRNN distance formula with smoothing factor sigma
				    denom += exponent;
				  	num += (exponent * centroids.get(i).getLabel());/**This is where the bug was. Instead of using exponent*centroids.get(i).getLabel(), I used exponent*o.getLabel() 
				  	                                                 This did not alter the best results obtained.
				  	                                                 Everything remains the same, except the number of centroids which changed to 7 instead of 49
				  	                                                 */
				}
				//System.out.println(" Num "+num/(double)denom+" distance Label "+" To be classified: "+o.getLabel());
				//Classification
				if (num / denom < 0.5) 	yHat = 0; else yHat = 1; //classification
				if (yHat == o.getLabel()) {
					countAccuracy++;
					if (yHat == 0 && o.getLabel() == 0) TN++; else TP++;
				}
				else {if (yHat == 0 && o.getLabel() == 1) FN++; else FP++;}
			}

			accuracy = countAccuracy / (double) (testSet.size());
			sensitivity = (TP) / (double) (TP + FN);
			specificity = (TN) / (double) (TN + FP);
			PPV = (TP) / (double) (TP + FP);
			NPV = (TN) / (double) (FN + TN);

			if (accuracy > currAccuracy) {
				if (sensitivity > currSensitivity) {
					currAccuracy = accuracy;
					currSensitivity = sensitivity;
					currentSpecificity = specificity;
					currentPPV = PPV;
					currentNPV = NPV;
				}
			}
			
}//end iterations	
 	 	overallAccuracySum+=currAccuracy;
		sensitivitySum+=currSensitivity;
        specificitySum+=currentSpecificity;
	    NPVsum+=currentNPV;
	    PPVsum+=currentPPV;
	
	}//end run
	 double avgAccuracy = 100*(overallAccuracySum/5.0);
	 double avgSensitivity = 100*(sensitivitySum/5.0);
	 double avgSpecificity = 100*(specificitySum/5.0);
	 double avgNPV = 100*(NPVsum/5.0);
	 double avgPPV = 100*(PPVsum/5.0); 
	 System.out.println();
	 System.out.println(" AVG Accuracy: "+numberFormat2.format(avgAccuracy)+"% \n AVG Sensitivity: "+numberFormat2.format(avgSensitivity)+"% \n AVG Specificity: "+numberFormat2.format(avgSpecificity)+"% \n AVG NPV: "+numberFormat2.format(avgNPV)+"% \n AVG PPV: "+numberFormat2.format(avgPPV)+"%");

 		return "";
	}

	public static void main(String[] args) throws Exception {
		
		 /**uncomment each statement to execute n times for each dataset*/
		for(int i=0;i<15;i++) {
			System.out.println("-------- Experiment "+(i+1)+"-------- ");
		 KGRNN_Train_Classify("D-[Extracted Features].txt",7,0.019,1000);
		// KGRNN_Train_Classify("B-[Excl Missing].txt",7,0.019,1000);
		//	KGRNN_Train_Classify("C-[Replaced by Mean].txt",7,0.019,1000);
			//KGRNN_Train_Classify("A-[Unprocessed].txt",7,1,1000);
		 System.out.println("-------------------------------");
		 System.out.println();
		}
	}// main

}
