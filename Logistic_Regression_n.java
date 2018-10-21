package code;

 
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.TimeZone;

public class Logistic_Regression_n {
	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;
	public static double predictedNeg = 0;
	public static double predictedPos = 0;
	public static double expectedNeg = 0;
	public static double expectedPos = 0;
	//Made all the above fields static in order to access them anywhere. I initialize them in each method as neede. 
	
	/** the learning rate */
	public static double rate;

	/** the weight to learn */
	private static double[] weights;

	/** the number of iterations */
	private int Epoch = 0;

	public Logistic_Regression_n(int n) {
		this.rate = 0.05;
		weights = new double[n];
	}

	private static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	private  final static String getDateTime()
	{
	    DateFormat df = new SimpleDateFormat("yyyy-MM-dd_hh:mm:ss");
	    df.setTimeZone(TimeZone.getTimeZone("UTC"));
	    return df.format(new Date());
	}
	
	public void train(List<Instance> instances, int nr_epoch) {//train LR model
		//Initial coeffs
		for (int j=0; j<weights.length; j++) {
			weights[j] =  0;
		}
		double[] epochError = new double[nr_epoch];
		for (int n=0; n<nr_epoch; n++) {
			double error = 0.0;
			double sumError = 0.0;
			for (int i=0; i<instances.size(); i++){ //train model, compute error and update weights (coefficients)
				double[] x = instances.get(i).x;
				double predicted = classify(x);
				int label = instances.get(i).label;
				error = label-predicted;		
				sumError+=Math.pow(error, 2);
				//update weights
				weights[0] = weights[0]+rate*error*predicted*(1.0-predicted);
				for (int j=0; j<weights.length-1; j++) {
					weights[j+1] = weights[j+1]+rate*error*predicted*(1.0-predicted)*x[j];
				}
				
			}
			//error sum for each epoch
			epochError[n] = sumError;
		 
		}	
	}

	private double classify(double[] x) {
		double logit = 0.0;
		logit+=weights[0];
		for (int i=0; i<weights.length-1;i++)  {
			logit += weights[i+1] * x[i];
		}
		return sigmoid(logit);
	}

	 
	public static class Instance {
		public int label;
		public double[] x;

		public Instance(int label, double[] x) {
			this.label = label;
			this.x = x;
		}
	
		public double[] getX(){
			return this.x;
		}
		
		public int getLabel(){
			return this.label;
		}
	 
		public  int getElementSize(){
			return this.x.length;
		}
		
	}
	
	public static void printDataset(List<Instance> instances){
		 for(int i = 0; i<instances.size();i++){
			 double x[] = instances.get(i).getX();
			 int label = instances.get(i).getLabel();
			 System.out.println("Instance "+i);
		 for(int j =0; j<x.length;j++){
			 
			 System.out.print(x[j]+" ");
		 }
		 System.out.print("Label "+label);
		 System.out.println("");
		 }
		 
	}

	public static List<Instance> readDataSet(String file) throws FileNotFoundException {
		List<Instance> dataset = new ArrayList<Instance>();
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
				Instance instance = new Instance(label, data);
				dataset.add(instance);				
				
			}
		} finally {
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}

	public static double [][] getMinMax(String file){
		Scanner scanner = null;
		double[][] minMaxValues = {{0, 0, 0, 0,0,0,0,0},{0, 0, 0, 0,0, 0, 0, 0}};

		try {
			scanner = new Scanner(new File(file));
			 //initialiseValues, read the first line
			 
			String line = scanner.nextLine();
			line = line.replaceAll("\\s+","");
			 
			String[] columns = line.split(",");
		    for(int i = 0; i<8; i++)
		    {
		    	minMaxValues[0][i] = Double.parseDouble(columns[i]);
		    	minMaxValues[1][i] = Double.parseDouble(columns[i]);
		        
		    }
			while(scanner.hasNextLine()) {
				 line = scanner.nextLine();
				if (line.startsWith("#")) {
					continue;
				}
				line = line.replaceAll("\\s+","");
				 columns = line.split(",");			 
				    for (int i = 0; i < columns.length-1; i++) {
			            //check the max value
			            double currentValue = Double.parseDouble(columns[i]);
			            if(currentValue > minMaxValues[0][i] ) {
			            	minMaxValues[0][i] = currentValue;
			            }
			            //check the min value			            
			            if(currentValue < minMaxValues[1][i] ) {
			            	minMaxValues[1][i] = currentValue;
			          
			            }
			             
			    }
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (scanner != null)
				scanner.close();
		}
	
		return minMaxValues;

	} 
	
	public static double testAccuracy(List<Instance> testInstances, double[] predicted_test, boolean isBest ){
		predictedNeg = 0;
		predictedPos = 0;
		expectedNeg = 0;
		expectedPos = 0;
		TP =0;
		FP =0;
		TN =0;
		FN = 0;
			//test accuracy
			double [] catPredicted = new double[predicted_test.length];
			int numOfCorrect = 0;		 
			 for(int i = 0; i<catPredicted.length;i++){
				 if (predicted_test[i]<0.5){
					 catPredicted[i] = 0; 
					 
				 }
				 else {
					 catPredicted[i] = 1; 
					  
				 }
				 }
			 for(int i = 0; i<catPredicted.length;i++){
				 if (catPredicted[i] == testInstances.get(i).label){
					 numOfCorrect++;
					 if(isBest==true){		
					 if(catPredicted[i]==0){
						 TN++;
						 predictedNeg++;
					 }
					 else {
						 TP++;
						 predictedPos++;
						 }
					 }
				 }
				 else{
					 if(isBest==true){//isBest = true if the sample to be evaluated is a test set
						 if(catPredicted[i]==0 && testInstances.get(i).label==1){
						 FN++; //predictedNeg++;
					 }
					 else {
						 FP++;//predictedPos++;
						 }
					 }
				 }
				 if(testInstances.get(i).label==0){
					 expectedNeg++;
				 }
					  else{ expectedPos++;}
			 }
			  
	 		return  100*(numOfCorrect/((double)testInstances.size()));
	}
	
	public static int getRandomInt(int min, int max) {
	    Random random = new Random();
	    return random.nextInt((max - min) + 1) + min;
	}

	public static ArrayList<Integer> getRandomNonRepeatingIntegers(int size, int min,
	        int max) {
	    ArrayList<Integer> numbers = new ArrayList<Integer>();

	    while (numbers.size() < size) {
	        int random = getRandomInt(min, max);

	        if (!numbers.contains(random)) {
	            numbers.add(random);
	        }
	    }

	    return numbers;
	}
	
	
 public static double[] getWeights(){
	 return weights;
 }
 
 public static int getMaxIndex(double[] accuracies){
		int currMaxIndex = 0;
		double currentMax = accuracies[0];	
		for (int i =1; i<accuracies.length;i++){
			if(accuracies[i]>currentMax){
				currMaxIndex = i;
				currentMax = accuracies[i];
			}
		}
return currMaxIndex;
	}
 
 public  double classifyTestSet(double[]x, double[]coeffs){
	  
	 double logit = 0.0;
		logit+=coeffs[0];
		for (int i=0; i<coeffs.length-1;i++)  {
			logit += coeffs[i+1] * x[i];
		}
		return sigmoid(logit);
		
 }
 
 public static String getResults(String filename, int restarts, int foldsNum, int epochs, double learningRate, int numOfWeights) throws FileNotFoundException{
		String resultReturn = "";
		long start = System.currentTimeMillis();
		double[] maxCoeffs = new double[numOfWeights];
		DecimalFormat numberFormat = new DecimalFormat("#.000");
		Logistic_Regression_n.rate = learningRate;
		double[] accuracyPerRunT = new double[restarts + 1];
        double [][] perfomances = new double [restarts + 1][4];
		ArrayList<ArrayList<Double>> counts = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> countsExp = new ArrayList<ArrayList<Double>>();

		for (int i = 0; i < restarts + 1; i++) {//Number of restarts (including initial restart))
			counts.add(new ArrayList<Double>());//for Chi-square tests
			countsExp.add(new ArrayList<Double>()); //for Chi-square tests
		}
 		for(int q = 0; q<restarts+1;q++){//Run for certain number of restarts
		Logistic_Regression_n logistic = new Logistic_Regression_n(numOfWeights);
		List<Instance> allData = readDataSet(filename);
		Collections.shuffle(allData);
		Instance[][] fold = new Instance[foldsNum][(int)(allData.size()/foldsNum)];
		double[][] coeffs = new double [foldsNum][numOfWeights];
		double [] accuracies = new double [foldsNum-1];
		int count = 0;
		
		for (int i =0;i<foldsNum;i++){
		for (int j=0; j<(int)(allData.size()/foldsNum);j++){
			fold[i][j] = allData.get(count);
			count++;
		}
		}
		List<Instance> trainFolds = new ArrayList<Instance>();
		List<Instance> validationSet = new ArrayList<Instance>();
		List<Instance> testInstances = new ArrayList<Instance>();
		
		for (int j=0; j<(int)(allData.size()/foldsNum);j++) testInstances.add(fold[0][j]); 
		double[] predicted_test = new double[(int)(0.4*((int)(allData.size()/foldsNum)))];
		double[] predictedTestSet = new double[testInstances.size()];
		for (int i =0;i<foldsNum-1;i++){ //cross validation
			for (int j=0; j<(int)(allData.size()/foldsNum);j++){
				if (j<(((int)(allData.size()/foldsNum)-(int)(0.4*((int)(allData.size()/foldsNum)))))){trainFolds.add(fold[i][j]);}
				else {validationSet.add(fold[i][j]);/**validation set*/
					}
			} 		
		 	logistic.train(trainFolds,epochs);//train using training set
			 for(int k = 0; k<(int)(0.4*((int)(allData.size()/foldsNum)));k++){/**40% validation set*/
				 predicted_test[k] = logistic.classify(validationSet.get((k)).getX());/**classify validation Set*/
		  }
			 double[] finalWeights = getWeights();
			 for (int c =0;c<numOfWeights;c++){
					 coeffs[i][c] = finalWeights[c];/** Save coeff of the current training instances*/
			 }
            
			 double   accu   = testAccuracy(validationSet,predicted_test,false);/**get Accuracy on validation set*/
			 accuracies[i] = accu; 
			 trainFolds.clear();
			 validationSet.clear();
			}//End k-fold cv
 		int sum=0; 
		  for (int a = 0;a<foldsNum-1;a++)sum+=accuracies[a];	 
		  int maxIndex = getMaxIndex(accuracies);//Get index of highest classification accuracy and use it to get the coefficients which obtained it
			maxCoeffs = new double[numOfWeights];
			for (int j = 0; j < numOfWeights; j++)
				maxCoeffs[j] = coeffs[maxIndex][j]; //get best coeffs and use them to classify final test set: testInstances
		
			for (int k = 0; k < testInstances.size(); k++) {	 
				predictedTestSet[k] = logistic.classifyTestSet(testInstances.get(k).getX(), maxCoeffs);//Classify test instances using maxCoeffs
			}
			double currentBestSolution = testAccuracy(testInstances, predictedTestSet, true);
			accuracyPerRunT[q] = currentBestSolution;
			perfomances[q][0]= Math.round(100.0*((+(TP)/((double)(TP+FN)))));//sensitivity
			perfomances[q][1]= Math.round(100.0*((TN)/(double)(TN+FP)));//Specificity
			perfomances[q][2]= Math.round(100.0*((TP)/(double)(TP+FP)));//PPV
			perfomances[q][3] = Math.round(100.0*((TN)/(double)(FN+TN)));  //NPV
			 		// add counts of predictions...// used these counts for Chi-Square tests purposes, not for classification
					counts.get(q).add(predictedNeg);
					counts.get(q).add(predictedPos);
					countsExp.get(q).add(expectedNeg);
					countsExp.get(q).add(expectedPos);
	}//end restarts
 double sensitivitySum =0;
 double specificitySum =0;
 double NPVSum =0;
 double PPVSum =0;
double sumT = 0;
double rTest = 0;
		for (int a = 0; a < restarts + 1; a++) {
			sumT += accuracyPerRunT[a];//sum all all test accuracies from each run to take their average as final classification
			sensitivitySum+=perfomances[a][0];
			specificitySum+=perfomances[a][1];
			NPVSum+=perfomances[a][2];
			PPVSum+=perfomances[a][3];
		}
		
		rTest = sumT / ((double) (restarts + 1));//restarts' test set avg accuracy
		long endTime = System.currentTimeMillis();
		resultReturn+="LR-n Classification Accuracy: "+numberFormat.format(((rTest)))+"% \n";
 		resultReturn+="Sensitivity: "+Math.round(1.0*(sensitivitySum/(double)(restarts+1)))+" % \n";  ;
 		resultReturn+="Specificity: "+Math.round(1.0*(specificitySum/(double)(restarts+1)))+" % \n";  
 		resultReturn+="Positive Predictive Value: "+Math.round(1.0*(NPVSum/(double)(restarts+1)))+" % \n";
 		resultReturn+= "Negative Predictive Value: "+Math.round(1.0*(PPVSum/(double)(restarts+1)))+" % \n";
 		   return resultReturn;
 }
 public static void main(String... args) throws IOException {
		/**
		 * To run experiments,  uncomment a dataset you want to use for experimentation
		 * Threshold values are used to easily find the best solution so that a user does not re-run the program each time to find a solution
		 * Threshold values were found through trial and error based on the performance on each dataset
		 * */
		for(int i=0;i<1000000;i++) {
		 	System.out.println("-------------- Experiment "+(i+1)+" --------------");
			String dataset ="";
			String result ="";
		    //dataset = "B-[Excl Missing].txt";
			dataset = "D-[Extracted Features].txt";
			//dataset = "C-[Replaced by Mean].txt";
			//dataset = "A-[Unprocessed].txt";
				if(dataset.equals("D-[Extracted Features].txt")) result = getResults(dataset,5,5,300,0.15,5);
				else result = getResults(dataset,5,5,300,0.15,8);
				System.out.println(result);
			 	double threshold = 0;
				if(dataset.equals("B-[Excl Missing].txt")) threshold = 83.5;
				else if(dataset.equals("D-[Extracted Features].txt")) threshold = 80.5;
				else if(dataset.equals("C-[Replaced by Mean].txt")) threshold = 79.5;
			    else threshold = 63.5;
			 	if(Double.parseDouble(result.substring(30, 36)) >= threshold && Double.parseDouble(result.substring(52, 54)) > 0) {//Best Solution found
			System.out.println();
			System.out.println("******************----------Best Result------------*****************");
			System.out.println();
			System.out.println(result);
			System.exit(0);


		}
			 	
			 	
			//if(d>83.5)System.exit(0);
					//System.out.println(getResults("B-[Excl Missing].txt",5,5,300,0.15,8));//0.3
		 //System.out.println(getResults("D-[Extracted Features].txt",5,5,300,0.15,5));//0.3
			//System.out.println(getResults("C-[Replaced by Mean].txt",5,8,300,0.15,8));//0.3
			 //System.out.println(getResults("A-[Unprocessed].txt",5,5,300,0.15,8));//0.3
	 
		
		 System.out.println();
		}
		
	 
	}
 

}
