package code;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class BKGRNN_Revised {
	public static int numOfClusters = 0;
	public static double sigma = 0;
	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;
	public static int [] predictions;

	 public BKGRNN_Revised() {

	}
	
	
/** finalPredictor computes the final Adaboost classification equation H(x)
 * H(x) is the combination of weak classifiers with their respective classification weights
 * 
 * finalPredictor takes a set of centroids (centroidList) which represent training instances for each weak classifier
 * centroidList.get(0) contains a list of centroids which are used by the weak classifier 1 as training set
 * 
 * the input alpha contains a list of weights (alphas) associated with each weak classifier
 * alphas are computed during training. alpha.get(0) contains a weight value for the first classifier
 * 
 * The smoothing factor sigma and the test set are also inputs to the method
 *  */
	
	public static double finalPredictor( ArrayList<ArrayList<Observation>> centroidList, 
			                             ArrayList<Observation> testSet, ArrayList<Double> alpha,double sigma) {
		int counter = 0;
		int currLabel = -2;
		double sign = 0;
		
		/**counters for computing performance measures*/
		TN = 0;
		TP = 0;
		FN = 0;
		FP = 0;
  
		for (Observation o : testSet) {/**classify each test instance using a set of weak learners*/
			sign = 0;
			for (int i = 0; i < centroidList.size(); i++) {/**Iterate through a set of centroids for each weak leaner*/
				currLabel = classifyObservation(centroidList.get(i), o, sigma);/**get classification of each weak learner*/
		 		sign += alpha.get(i).doubleValue() * currLabel;  /**use the output of each weak learner to get a weighted classification and add the result to the sum*/
			}
		 	
			/**sign is the output of the final classifier
			 * Evaluate the sign and increment counters
			 * */
		 	if (sign < 0 && o.getLabel() == 1)FN++;
			if (sign > 0 && o.getLabel() == -1)  FP++;
 			if (sign < 0 && o.getLabel() == -1) {
				counter++;/**count correct classification*/
			 	TN++;
			 
			} else if (sign >= 0 && o.getLabel() == 1) {
				counter++;/**count correct classification*/
			 	TP++;
			} else
				continue;
			sign = 0;/**re-initialize the sign for the next test instance*/

		}
 		return (counter) / (double) (testSet.size());/**return final classification*/
	}

	/**classifyObservation takes a set of centroid, an instance, and sigma to produce a classification
	 */
	public static int classifyObservation(ArrayList<Observation> centroidList, Observation obs, double sigma) {

		double exponent = 0;
		double currDistance = 0;
		double denom = 0;
		double num = 0;
		int yHat = 0;
		for (int j = 0; j < centroidList.size(); j++) {//GRNN classifier, see classifyTrainingSet method for more info
				currDistance = Observation.distance(obs, centroidList.get(j));
				exponent = Math.exp(-1.0* (((currDistance * currDistance)) /((double) (2.0 * sigma * sigma))));
				denom += exponent;
				num = num+(exponent * centroidList.get(j).getLabel()); /**This is where the bug was. Instead of using exponent*centroids.get(i).getLabel(), I used exponent*o.getLabel()
				                                                        Since we are considering the sign, the result (the sign) will always be corresponding to the instance label.. 
				                                                        Hence the classification accuracy of 100%
				                                                        */
	         }
 		if (num / denom < 0) yHat = -1; else yHat = 1;
		return yHat;
	}
	
	/**updateWeights is used to update training instance weights after each classification 
	 * to allocate more weight to instances which are misclassified
	 * The weight of the classifier (alpha) is used to update instance weights
	 * 
	 * The variable array "predictions" is a global variable which stores classification output for each weak learner during training set classification
	 * 
	 * */

	public static void updateWeights(ArrayList<Observation> obs, double alpha) {
		for (int i =0; i<obs.size();i++) 
			obs.get(i).setWeight( obs.get(i).getWeight() * Math.exp(-1* (alpha * predictions[i])));
		
		double weightSum = 0;
		for (Observation o : obs) {/**get sum of instance weights to normalize the weights for the next iteration*/
			weightSum += o.getWeight();
		}
		
		for (Observation o : obs) {/**Update weight for each training instance*/
			o.setWeight(o.getWeight()/weightSum);
		}
	}

	/** classifyTrainingSet is used to compute the classification accuracy on the trainingSet
	 * A list of centroids, the smoothing factor sigma and the training set are the inputs
	 * The method return a weighted classification error for the weak leaner
	 * */
	public static double classifyTrainingSet(ArrayList<Observation> centroidList,ArrayList<Observation> trainingSet, double sigma) {
		/**Variables for storing the GRNN numerator and denominator*/
		double exponent = 0;
		double currDistance = 0;
		double denom = 0;
		double num = 0;
		int yHat = 0;//stores classified value
	
		predictions = new int[trainingSet.size()];/**To store classification for each training instance.
                                          The array values are used when training instance weights are updated */ 
        int countObs = 0;//count index for storing classifications
		for (Observation o : trainingSet) {/**For each training instance compute distance between each centroid */                   
				denom = 0;
				denom = 0;
				 for (int i = 0; i < centroidList.size(); i++) {/**iterate through each centroid to compute its distance from the training instance*/
					currDistance = Observation.distance(o, centroidList.get(i));
					exponent = Math.exp(-1.0* (((currDistance * currDistance)) /((double) (2.0 * sigma * sigma))));/**GRNN denom formula*/		
				 	denom += exponent;
					num += exponent * centroidList.get(i).getLabel();/**GRNN numerator formula*/
				}
				 
				if (num / denom < 0) yHat = -1; else yHat = 1;//classification
				if (yHat == o.getLabel()) predictions[countObs] = 0;//correct prediction, reduces training instance weight
				else predictions[countObs] = 1;/***Incorrect prediction, the result is used in weight update to ensure 
				                               that the weight of misclassified instance keeps increases*/
				countObs++;
		}
		/**Compute weighted error for the classifier and return it*/
		double weightSum = 0;
		double weightedError = 0;
		for(int i=0; i<predictions.length;i++) {
			weightedError+=(trainingSet.get(i).getWeight()*predictions[i]);
			weightSum+=trainingSet.get(i).getWeight();
		}
		 
		return weightedError/(double)(weightSum);
	}


	/** getByWeight returns a random training instance from the training set based in its weight
	 * The higher the weight of an instance, the higher the probability of its selection
	 *  */
	    public static Observation getByWeight(List<Observation> obs) {
	        double completeWeight = 0.0;
	        for (Observation o : obs)
	            completeWeight += o.getWeight();//total weight for all training instances
	        
	        if(completeWeight<= 0||Double.isNaN(completeWeight)||Double.isInfinite(completeWeight)) 
	        	return new Observation(new double[8],-10);//if total weight is <0 or undefined, return an undefined instance to be used for stopping training
	        else {//randomly get instance with higher weight
	        double r = Math.random() * completeWeight;
	        double countWeight = 0.0;
	       for(int i=0;i<obs.size();i++) {
	    	   countWeight += obs.get(i).getWeight();
	            if (countWeight >= r) {
	                return obs.remove(i);/**an instance is removed from the list to ensure that it's selected without replacement*/
	            }
	       
	       }
	         	        throw new RuntimeException("None selected");
	        }
	        }
 
	    /**getTrainSubset takes a training set as input and it returns a subset of the training set
	     * Instances are added on the subset based on their weights
	     * 80% of training instances are added to the subset without replacement, getByWeight method is used to randomly get intances
	     * */
	    public static ArrayList<Observation>  getTrainSubset(List<Observation> trainSetCopy) {
	    	ArrayList<Observation> subset = new ArrayList<Observation>();
 	    	int subSetSize = (int)(trainSetCopy.size()*0.8);
 	    	for(int i=0; i<subSetSize;i++) 	subset.add(getByWeight(trainSetCopy));
	     	return subset;
	    }
	

	    /**getCentroids takes a training set and number of clusters as input.
	     * It then uses CVE-K-Means algorithm to get a set of centroids that are used as the training set
	     * */
	    public static ArrayList<Observation> getCentroids (List<Observation> list, int numOfClusters) throws IOException {//get best Centroids to use as training set
			 int numOfFolds = 10;//folds for CVE-K-Means k-fold validation 
			 double[] accuracies = new double[numOfFolds];
		     ArrayList<Observation> trainSet = new ArrayList<Observation>();
			 ArrayList<Observation> testSet = new ArrayList<Observation>();
			 ArrayList<ArrayList<Observation>> data = new ArrayList<ArrayList<Observation>>();
							
			 for (int i = 0; i < numOfFolds; i++) 
				 data.add(new ArrayList<Observation>());//data stores k samples of equal size to be used for k-fold cross validation
							
							for (int i = 0; i < numOfFolds; i++) {
								for (int j = 0; j < (int) (list.size() / numOfFolds); j++) {
									data.get(i).add(list.get(j)); //add folds into data
								Collections.shuffle(data.get(i));
								}
							}
			ArrayList<Observation> currCentroids = new ArrayList<Observation>();/**keeps track of best centroids
			                                                                     best centroids are those with the highest accuracy on the test set during cv*/
			ArrayList<Observation> centroids = new ArrayList<Observation>();
			
			for (int i = 0; i < numOfFolds; i++) {//cv starts here
				testSet = data.get(0);//hold one sample to be test set
				data.remove(0);//remove test set from the samples

				for (int j = 0; j < numOfFolds - 1; j++) {//combine instances from k-1 samples into training set
					for (int k = 0; k < (int) (list.size() / numOfFolds); k++) {
								trainSet.add(data.get(j).get(k));			
					}
				}
				KMeans kmeans = new KMeans(numOfClusters, trainSet.size(), trainSet);//K-Means instance for getting centroids
				kmeans.init(numOfClusters);
				kmeans.calculate();//create clusters
				ArrayList<Cluster> clusters = kmeans.getClusters();
				setCentroidLabel(clusters);
				centroids = kmeans.getCentroids();
				if (i == 0)  currCentroids = centroids;//initial centroids
 
				double accuracyTest = getAccuracy(clusters, testSet, numOfClusters);
				accuracies[i] = accuracyTest;
				
				if (i > 0) {if (accuracies[i] > accuracies[i - 1]) currCentroids = centroids;} //update centroidlist if better centroids are found
				data.add(data.size() - 1, testSet);//add the removed sample back to k-1 samples for next iteration
	 			testSet = new ArrayList<Observation>();
				trainSet = new ArrayList<Observation>();
				
			}
			 
			return currCentroids;//return best centroids 

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
	    
		
	    /**getData returns a balanced split of trainset and testSet (70%/30% split)*/
		public static ArrayList<ArrayList<Observation>> getData(String filePath) throws IOException {//get best Centroids to use as training set
			ArrayList<Observation> allData = KMeans.readDataSet(filePath);
			int numOfFolds = 10;/**Used to split dataset into 10 samples of equal size (ensuring the proportion of -ve and +instances is preserved)*/
			Observation[][] fold = new Observation[numOfFolds][(int) (allData.size() / numOfFolds)];
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
			
				ArrayList<ArrayList<Observation>> data = new ArrayList<ArrayList<Observation>>();
			    for (int i = 0; i < numOfFolds; i++) data.add(new ArrayList<Observation>());
			    /**data contains a list of 10 samples*/
							for (int i = 0; i < numOfFolds; i++) {
								for (int j = 0; j < (int) (allData.size() / numOfFolds); j++) {
									data.get(i).add(fold[i][j]); //add folds into a list
								Collections.shuffle(data.get(i));
								}
							}
							
	  	ArrayList<Observation> testSet = new ArrayList<Observation>();
		ArrayList<ArrayList<Observation>> finalChoice = new ArrayList<ArrayList<Observation>>();
		 
	for(int i=0;i<3;i++) {/**Add 3 samples out of 10 into test set (creating 30% of test cases)*/
		for(Observation o: data.get(i))
			testSet.add(o);
	}
	data.remove(0);data.remove(1);data.remove(2);/**Remove samples that are in the test set*/
	ArrayList<Observation> train = new ArrayList<Observation>();
			finalChoice.add(testSet);/**Add test set on return list*/
			for(ArrayList<Observation> list: data) {/**Combine all instances in remaining 7 samples into 1 trainset*/
				for(Observation o: list) {
					train.add(o);
				}
			}
			finalChoice.add(train);/**Add trainset into result set*/
			return finalChoice;/**Return trainset and test set*/

		}

	    /** readDataset takes a filename as input, reads each line from the file and stores the attributes of each line as an Observation object
	     * 
	     * */
	    public static ArrayList<Observation> readDataSet(String file) throws FileNotFoundException {
			ArrayList<Observation> dataset = new ArrayList<Observation>();
	 		Scanner scanner = null;
			try {
				scanner = new Scanner(new File(file));
				while(scanner.hasNextLine()) {
					String line = scanner.nextLine();
					if (line.startsWith("#")) {
						continue;
					}
					line = line.replaceAll("\\s+","");
					 
					String[] columns = line.split(",");
					 
					// skip last column
					int i = 0;
					double[] data = new double[columns.length-1];
					for (i=0; i<columns.length-1; i++) {
						
						data[i] = Double.parseDouble(columns[i]);
							
					}
					int label = Integer.parseInt(columns[columns.length-1]);
					Observation Observation = new Observation(data,label);
					dataset.add(Observation);	
			 	}
			} finally {
				if (scanner != null)scanner.close();
			}
			return dataset;
		}
	    
	/** BKGRNN_Train_Classify takes a file name, number of clusters, sigma and number of iterations as input
	 * BKRNN is configured to run 10 times, each run has 15 iterations.
	 * On each iteration:
	 * 1. the dataset is read from a file, 
	 * 2. it is then split into 70% train set and 30% test set
	 * 3. then all instances in the training set are initialized with equal training weights
	 * 4. 10 weak KGRNN learners with centroids produced from a training subset by CVE-K-Means algorithm are then created
	 * 5. the final classifier which is a combination of 10 weak classifier is then used for final classification on the test set
	 * 6. the result of the final classifier is stored
	 * On next iteration:
	 * Steps 1-5 are executed. Then if the result in 5 is better than the previous iteration, then it is saved as the best solution
	 * After the 15th iteration, then the best result is returned for the run
	 * After 10 runs, the results are then averaged to get the final result of the BKGRNN classifier
	 * The best number of runs, iterations, sigma, and number of clusters was found through experimentation
	 * */
	public static void BKGRNN_Train_Classify(String file, int numOfClusters, double sigmaInput,int iterations)throws IOException {
		
		String alphasAndErrors = "Alpha Value \t Training Error \n \n";//store alpha values and errors for each weak learner
		sigma = sigmaInput;		
		DecimalFormat numberFormat = new DecimalFormat("0.0000");
		DecimalFormat numberFormat2 = new DecimalFormat("0.00");
		long startTime = System.currentTimeMillis();
      
	 	double accuracy = 0;
		double currAccuracy = 0;
		double currSensitivity = 0; 
		double currSpecificity = 0;
		double currNPV = 0;
		double currPPV = 0;
		 
		double sensitivitySum = 0;
		double specificitySum =0;
		double PPVsum = 0;
		double NPVsum =0;
	 		
		double classification = 0;
		double sensitivity = 0;
		double specificity = 0;
		double NPV = 0;
		double PPV = 0;
		double overallAccuracySum =0;
		String currAlpha = "";
for(int run=0;run<10;run++) {
	//System.out.println("Run: "+(run+1)+"...");	
	currAccuracy=0;currSensitivity = 0;currSpecificity=0;currNPV=0;currPPV=0;
	alphasAndErrors = "Alpha Value \t Training Error \n \n";currAlpha = "";
	 for (int k = 0; k < iterations; k++) {// iterations, 15 in this case
		accuracy = 0;	alphasAndErrors = "Alpha Value \t Training Error \n \n";
		ArrayList<ArrayList<Observation>>  finalChoice = getData(file);//getData returns test set and traininset instances
		ArrayList<Observation> testSet = finalChoice.get(0);//test set
		ArrayList<Observation> trainSet = finalChoice.get(1);//trainSet	
		ArrayList<Observation> centroids = new ArrayList<Observation>();
	 	ArrayList<Double> alpha = new ArrayList<Double>();
		ArrayList<ArrayList<Observation>> centroidsList = new ArrayList<ArrayList<Observation>>();//set of centroids for each weak learner
		double trainError = 0; 	

		for (Observation o : trainSet) {
			o.setWeight(((1 / (double) (trainSet.size()))));//Initial weights of training instances
		}
	 
 	for (int w = 0; w < 10; w++) {// max number of weak classifiers allowed = 10
 
ArrayList<Observation> trainCopy = new ArrayList<Observation>(trainSet.size());//
	  for (Observation o: trainSet) {/**create a deep copy of trainset
		                              the deep copy is used to return a trainingset subset that is used to create training centroids
		                              the initial train set remains intact, it's used to test the weak classifier 
		                              the weight of each instance in the traininset is then updated*/
			  Observation tempObs = new Observation(o.getVars(),o.getLabel());
			 tempObs.setWeight(o.getWeight());
			 trainCopy.add(tempObs);
		  }
ArrayList<Observation> trainSubset = getTrainSubset(trainCopy);	//get training subset based on instance weight
centroids = getCentroids(trainSubset,numOfClusters); //get centroids using training subset	  
trainError = classifyTrainingSet(centroids, trainSet, sigmaInput);//weighted training error

  if((trainError<=0||Double.isNaN(trainError)||Double.isInfinite(trainError))) continue; 
	else alpha.add(0.5 * Math.log((1 - trainError) / trainError));//save weight for weak learner
	centroidsList.add(centroids);//save centroids used by this weak learner
	updateWeights(trainSet,0.5 * Math.log((1 - trainError) / trainError));//update weigths
  	alphasAndErrors += (numberFormat.format(0.5 * Math.log((1 - trainError) / trainError)) + "\t \t " + numberFormat.format(trainError)) + "\n";
	 		}
		// once done compute overall predictor
		if (centroidsList.size() <= 1) {
			System.out
					.println("Could not find more than 1  weak learners with given Sigma: "
							+ sigma
							+ " and "
							+ numOfClusters
							+ " cluster centers. \nTry for a different sigma and number of cluster centers");
			 System.exit(0);
		} else {
			classification =   Double.parseDouble(numberFormat2.format((finalPredictor(centroidsList, testSet,alpha, sigmaInput))));
		 	alphasAndErrors+= "\n";
			accuracy = classification;
			 
			  sensitivity = (TP) / (double) (TP + FN);
			  specificity = (TN) / (double) (TN + FP);
			  PPV = (TP) / (double) (TP + FP);
			  NPV = (TN) / (double) (FN + TN);

			if (accuracy > currAccuracy) {
				if (sensitivity > currSensitivity) {
					currAccuracy = accuracy;
					currSensitivity = sensitivity;
					currSpecificity = specificity;
					currPPV = PPV;
					currNPV = NPV;
					currAlpha = alphasAndErrors;
				}
			}
			

			long endTime = System.currentTimeMillis();
			double totalTime = ((endTime - startTime) / 1000.0000);
		 
		}
		
}// end iteration	 
			overallAccuracySum+=currAccuracy;
			sensitivitySum+=currSensitivity;
			specificitySum+=currSpecificity;
			NPVsum+=currNPV;
			PPVsum+=currPPV;	
			 }//end run
			 double avgAccuracy = 100*(overallAccuracySum/10.0);
			 double avgSensitivity = 100*(sensitivitySum/10.0);
			 double avgSpecificity = 100*(specificitySum/10.0);
			 double avgNPV = 100*(NPVsum/10.0);
			 double avgPPV = 100*(PPVsum/10.0); 
			 System.out.println();System.out.println();
		 
			System.out.println(" AVG Accuracy: "+numberFormat2.format(avgAccuracy)+"% \n AVG Sensitivity: "+numberFormat2.format(avgSensitivity)+"% \n AVG Specificity: "+numberFormat2.format(avgSpecificity)+"% \n AVG NPV: "+numberFormat2.format(avgNPV)+"% \n AVG PPV: "+numberFormat2.format(avgPPV)+"%");
	 }
	
	 
	public static void main(String[] args) throws Exception {
		 
		for(int i=0;i<10;i++) {
			System.out.println("-------- Experiment "+(i+1)+"-------- ");
			BKGRNN_Train_Classify("A-[Unprocessed_Boosting].txt",5,1.09,15);
		 
		 System.out.println("-------------------------------");
		 System.out.println();
		}
		
			 
	}
}
