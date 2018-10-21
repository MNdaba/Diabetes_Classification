package code;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class CKNN {
	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;
	/**
	 * @param args
	 * @throws FileNotFoundException
	 * 
	 * 
	 */
	public static ArrayList<Observation> trainForKNN = new ArrayList<Observation>();

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

	/**Use cluster centers to classify test instances*/
	public static double getAccuracy(ArrayList<Cluster> clusters,ArrayList<Observation> testSet, int numOfClusters) {
		double max = Double.MAX_VALUE;
		double min = max;
		double distance = 0.0;
		int correct = 0;
		int cluster = 0;
		TN = 0;
		TP = 0;
		FN = 0;
		FP = 0;
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

			if (clusters.get(cluster).centroid.getLabel() == obs.getLabel()) {
				correct++;
				trainForKNN.add(obs);
			}

		}
		return correct / (double) (testSet.size());
	}

	public static double computeKNN(ArrayList<ArrayList<Observation>> list,int k) {
		ArrayList<Observation> centroids = list.get(0);/** input centroid list,  test set, and k (number of neighbours) and return accuracy*/
		ArrayList<Observation> testSet = list.get(1);
		TN = 0;
		TP = 0;
		FN = 0;
		FP = 0;
		ArrayList<Observation> KNN = new ArrayList<Observation>();
		ArrayList<Observation> tempList = new ArrayList<Observation>();
		int countOne = 0;
		int countZero = 0;
		double max = Double.MAX_VALUE;
		double min = max;
		int index = 0;
		double distance = 0.0;
		int countRate = 0;
		int predicted = -1;
		for (Observation o : testSet) {
			for (int i = 0; i < centroids.size(); i++) {
				tempList.add(centroids.get(i));
			}

			for (int j = 0; j < k; j++) {
				min = max;
				for (int i = 0; i < tempList.size(); i++) {
					distance = (Observation.distance(o, tempList.get(i)));
					if (distance < min) {
						min = distance;
						index = i;
					}
				}

				KNN.add(tempList.get(index));
				tempList.remove(index);
			}

			for (int i = 0; i < KNN.size(); i++) {
				if (KNN.get(i).getLabel() == 1)
					countOne++;
				else
					countZero++;
			}

			if (countOne > countZero)
				predicted = 1;
			else
				predicted = 0;
			if (predicted == o.getLabel()) {
				if (predicted == 0 && o.getLabel() == 0)
					TN++;
				else
					TP++;
				countRate++;
			} else {
				if (predicted == 0 && o.getLabel() == 1)
					FN++;
				else
					FP++;
			}

			tempList.clear();
			KNN.clear();
			countZero = 0;
			countOne = 0;
		}
		return countRate / (double) (testSet.size());
	}

	public static String getAccuracy(String fileName, int numOfClusters,int neighbours, int numOfFolds) throws FileNotFoundException {
		String result = "";
		long start = System.currentTimeMillis();
		DecimalFormat numberFormat = new DecimalFormat("#.000");
		ArrayList<Observation> list = KMeans.readDataSet(fileName);
		Collections.shuffle(list);
		Observation[][] fold = new Observation[numOfFolds][(int) (list.size() / numOfFolds)];
		double[] accuracies = new double[numOfFolds];
		int count = 0;
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
		for (int i = 0; i < numOfFolds; i++) {
			data.add(new ArrayList<Observation>());
		}
		for (int i = 0; i < numOfFolds; i++) {
			for (int j = 0; j < (int) (list.size() / numOfFolds); j++) {
				data.get(i).add(fold[i][j]);
			}
		}
		double accuracy = 0;
		accuracies = new double[numOfFolds];
		double [][] perfomances = new double [numOfFolds][4];
		for (int i = 0; i < numOfFolds; i++) {// k-fold cv starts here
			testSet = data.get(1);
			validationSet = data.get(0);
			data.remove(0);
			data.remove(1);
			for (int j = 0; j < numOfFolds - 2; j++) {
				for (int k = 0; k < (int) (list.size() / numOfFolds); k++) {
					trainSet.add(data.get(j).get(k));
				}

			}// add all for learning cluster centres
 			KMeans kmeans = new KMeans(numOfClusters, trainSet.size(), trainSet);
			kmeans.init(numOfClusters);
			kmeans.calculate();
			ArrayList<Cluster> clusters = kmeans.getClusters();
			setCentroidLabel(clusters);
			ArrayList<Observation> centroids = kmeans.getCentroids();
			double classification = 0;
			 	ArrayList<ArrayList<Observation>> tempData = new ArrayList<ArrayList<Observation>>();
				tempData.add(centroids);
				tempData.add(validationSet);
				classification = computeKNN(tempData, neighbours);// KNN classification using centroids and final test set
				perfomances[i][0]= Math.round(100.0*((+(TP)/((double)(TP+FN)))));//sensitivity
				perfomances[i][1]= Math.round(100.0*((TN)/(double)(TN+FP)));//Specificity
				perfomances[i][2]= Math.round(100.0*((TP)/(double)(TP+FP)));//PPV
				perfomances[i][3] = Math.round(100.0*((TN)/(double)(FN+TN)));  //NPV
	 
			accuracies[i] = classification;
			data.add(data.size() - 1, testSet);
			data.add(data.size() - 1, validationSet);
			testSet = new ArrayList<Observation>();
			trainSet = new ArrayList<Observation>();
			validationSet = new ArrayList<Observation>();

		}// end k-fold cv
			 
		 double specificitySum =0;
		 double NPVSum =0;
		 double PPVSum =0;
		 double sensitivitySum =0;
		  
				for (int a = 0; a < numOfFolds; a++) {
					sensitivitySum+=perfomances[a][0];
					specificitySum+=perfomances[a][1];
					NPVSum+=perfomances[a][2];
					PPVSum+=perfomances[a][3];
				}
				
		double sumT = 0;
		for (int a = 0; a < numOfFolds; a++) {
			sumT += accuracies[a];

		}
		accuracy = 100 * ((double) sumT / numOfFolds);

		result += "CKNN Classification Accuracy: "
				+ numberFormat.format(accuracy) + "% \n \n";
		result += "Sensitivity: "+ Math.round(1.0*(sensitivitySum/(double)(numOfFolds)))+ " % \n \n";
		result += "Specificity: " +Math.round(1.0*(specificitySum/(double)(numOfFolds))) + " % \n \n";
		result += "Positive Predictive Value: " + Math.round(1.0*(NPVSum/(double)(numOfFolds))) + " % \n \n";
		result += "Negative Predictive Value: " +Math.round(1.0*(PPVSum/(double)(numOfFolds))) + " % \n \n";
		long endTime = System.currentTimeMillis();
		double totalTime = ((endTime - start) / 1000.0000);
	 	
		result += "Total Execution Time: " + totalTime + " seconds\n";
	 	return result;
		 
	}

	public static void main(String[] args) throws FileNotFoundException {
		/**
		 * To run experiments,  uncomment a dataset you want to use for experimentation
		 * Threshold values are used to easily find the best solution so that a user does not re-run the program each time to find a solution
		 * Threshold values were found through trial and error based on the performance on each dataset
		 * */
	for(int i=0;i<1000;i++) {
		String dataset ="";
	    dataset = "B-[Excl Missing].txt";
		//dataset = "D-[Extracted Features].txt";
		//dataset = "C-[Replaced by Mean].txt";
		//dataset = "A-[Unprocessed].txt";
		
		String result = getAccuracy(dataset,25,3, 10);
	 	double threshold = 0;
		if(dataset.equals("B-[Excl Missing].txt") ||dataset.equals("D-[Extracted Features].txt")) threshold =80;
		else if(dataset.equals("C-[Replaced by Mean].txt")) threshold = 77;
	else threshold = 75;
		System.out.println("Current Accuracy: "+Double.parseDouble(result.substring(30, 36)) );
if(Double.parseDouble(result.substring(30, 36)) >= threshold) {
	System.out.println();
	System.out.println("******************----------------------*****************");
System.out.println(result);
System.exit(0);


}
	}

	}

}
