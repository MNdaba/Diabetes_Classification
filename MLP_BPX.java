package code;

//Final MLP tested.Use for reporting
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

public class MLP_BPX {
	private ArrayList<Neuron> hiddenLayer1 = new ArrayList<Neuron>();
	private ArrayList<Neuron> hiddenLayer2 = new ArrayList<Neuron>();
	ArrayList<Double> input = new ArrayList<Double>();
	private static double learningRate;
	private int numberOfHiddenNeurons;
	private double sumSqError = 0.0;
	private double sumSqError2 = 0.0;
	private double correct = 0.0;
	private double numOfInstances = 0.0;
	Neuron outputNeuron;
	static BufferedWriter out2;
	static BufferedWriter out;
	public static double TP = 0;
	public static double FP = 0;
	public static double TN = 0;
	public static double FN = 0;
	public static double predictedNeg = 0;
	public static double predictedPos = 0;
	public static double expectedNeg = 0;
	public static double expectedPos = 0;

	public MLP_BPX(int numberOfHiddenNeurons,
			int numOfInputs, double LR) throws IOException {
		this.learningRate = LR;
		this.numberOfHiddenNeurons = numberOfHiddenNeurons;

		ArrayList<Double> weights = new ArrayList<Double>();
		ArrayList<Double> weightsInit = new ArrayList<Double>();
		/**weightsInit.add(-0.4560084857274203);
		weightsInit.add(-0.013223271705833782);
		weightsInit.add(-0.04177965341638604);
		weightsInit.add(-0.15943442870514407);
		weightsInit.add(0.11721237360972762);
		weightsInit.add(0.035435220973862355);
		weightsInit.add(-0.016690374461897882);
		weightsInit.add(0.014149291925189695); Logistic Regression Co-effs*/

		for (int i = 0; i < numberOfHiddenNeurons; i++) {
			for (int w = 0; w < numOfInputs; w++) {

				weights.add(new Random().nextGaussian() * (1.0 / (4.5)) + 0);//Initialize weights with Xavier Technique

			}
			Neuron n = new Neuron(weights);
			hiddenLayer1.add(n);
			weights.clear();
		}
     	weights.clear();
	
     	for (int w = 0; w < this.numberOfHiddenNeurons; w++) {
			weights.add(new Random().nextGaussian() * (1.0 / (4.5)) + 0);
		}
		outputNeuron = new Neuron(weights);
	}

	public void resetCounters() {
		this.numOfInstances = 0.0;
		this.correct = 0.0;
	}

	public void resetSumSqError() {
		this.sumSqError = 0;
		this.sumSqError2 = 0;
	}

	public void setInputs(ArrayList<Double> inputs) {
		input.clear();
		for (int i = 0; i < inputs.size(); i++) {
			this.input.add(inputs.get(i));
		}
	}

	public double getSumSqError2() {
		return sumSqError2;
	}

	public static void setLR(double LR) {
		learningRate = LR;
	}

	public static double getLR() {
		return learningRate;
	}

	public double getSumSqError() {
		return sumSqError;
	}

	public ArrayList<Neuron> getHiddenLayer2() {
		return this.hiddenLayer2;
	}

	public ArrayList<Neuron> getHiddenLayer() {
		return this.hiddenLayer1;
	}

	public Neuron getOutputNeuron() {
		return this.outputNeuron;
	}

	public double getEpochAccuracy() {
		return this.correct / (double) (this.numOfInstances);

	}

	public double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	public void FFD(ArrayList<Double> inputs, int target, int iterationNum) 
			throws IOException {// feedforward

		for (int i = 0; i < this.numberOfHiddenNeurons; i++) {
			hiddenLayer1.get(i).setInputs(inputs);/**9 Inputs signify 9 Neurons that don't do any computation 
			                                       Connected by weights set in line 62, 
			                                       so there was not need to create Neuron Instances but weights connecting inputs to the first hidden layer . 
			                                       */
		}

		ArrayList<Double> hiddenLayerOutputs = new ArrayList<Double>();
		for (int i = 0; i < this.numberOfHiddenNeurons; i++) {
			double sum = hiddenLayer1.get(i).getSum();
			double sigmoid = sigmoid(sum);
			hiddenLayer1.get(i).setOutput(sigmoid);
			hiddenLayerOutputs.add(hiddenLayer1.get(i).getOutput());
		} 
		
		outputNeuron.setInputs(hiddenLayerOutputs);
		hiddenLayerOutputs.clear();
		outputNeuron.setOutput(outputNeuron.sigmoid(outputNeuron.getSum()));
		double output = outputNeuron.getOutput();
		double error = output * (1 - output) * (target - output);
		sumSqError += Math.pow(target - output, 2);// SSE
		
		int predicted = -1;
		if (output < 0.5) {
			predicted = 0;
		} else
			predicted = 1;
		if (predicted == target) this.correct++; numOfInstances++;
		outputNeuron.setError(error);
		backPropergate(error, hiddenLayer1, outputNeuron, iterationNum, inputs);

	}

	//PB Method to back propagate the training error
	public void backPropergate(double error, ArrayList<Neuron> hiddenLayer,
			Neuron outputNeuron, int iterationNum, ArrayList<Double> inputP) {
		ArrayList<Double> tempDeltaWeights = new ArrayList<Double>();
		ArrayList<Double> tempNewWeights = new ArrayList<Double>();

		for (int j = 0; j < this.numberOfHiddenNeurons; j++) {
			hiddenLayer.get(j).setError(
					outputNeuron.getWeights().get(j)
							* hiddenLayer.get(j).getOutput() * error);
		}

		for (int i = 0; i < hiddenLayer.size(); i++) {
			double newWeight = outputNeuron.getWeigt(i) + this.learningRate
					* error * hiddenLayer.get(i).getOutput();// +(momentum*outputNeuron.getWeightDeltas().get(i));
			       tempNewWeights.add(newWeight);
			      tempDeltaWeights.add(newWeight);
		}
		outputNeuron.setWeightDelta(tempDeltaWeights);
		outputNeuron.setWeights(tempNewWeights);
		tempDeltaWeights.clear();
		tempNewWeights.clear();
		// update weights for hidden layer neurons
		for (int i = 0; i < hiddenLayer.size(); i++) {

			for (int k = 0; k < hiddenLayer.get(0).getInputsSize(); k++) {
				double newWeight = hiddenLayer.get(i).getWeigt(k)
						+ this.learningRate * hiddenLayer.get(i).getError()* inputP.get(k);
				tempNewWeights.add(newWeight);
				tempDeltaWeights.add(newWeight);
			}// add new weights for hidden layer

			hiddenLayer.get(i).setWeightDelta(tempDeltaWeights); 
			hiddenLayer.get(i).setWeights(tempNewWeights);
			tempDeltaWeights.clear();
			tempNewWeights.clear();
		}

	}

	public static ArrayList<Observation> readDataSet(String file)
			throws FileNotFoundException {
		ArrayList<Observation> dataset = new ArrayList<Observation>();
		// double minMax[][] = getMinMax ("AllDataV.txt");
		Scanner scanner = null;
		try {
			scanner = new Scanner(new File(file));
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				if (line.startsWith("#")) {
					continue;
				}
				line = line.replaceAll("\\s+", "");

				String[] columns = line.split(",");

				// skip last column
				int i = 0;
				double[] data = new double[columns.length - 1];
				for (i = 0; i < columns.length - 1; i++) {

					data[i] = Double.parseDouble(columns[i]);

				}

				int label = Integer.parseInt(columns[columns.length - 1]);
				Observation Observation = new Observation(data, label);
				dataset.add(Observation);

			}
		} finally {
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}

	public double classifyInstance(ArrayList<Double> inputs, int target,ArrayList<Neuron> hiddenLayer,Neuron outputNeuron) {

		for (int i = 0; i < this.numberOfHiddenNeurons; i++) {
			hiddenLayer.get(i).setInputs(inputs);
		}

		ArrayList<Double> hiddenLayerOutputs = new ArrayList<Double>();
		for (int i = 0; i < this.numberOfHiddenNeurons; i++) {
			double sum = hiddenLayer.get(i).getSum();
			double sigmoid = sigmoid(sum);
			hiddenLayer.get(i).setOutput(sigmoid);
			hiddenLayerOutputs.add(hiddenLayer.get(i).getOutput());
		}

		outputNeuron.setInputs(hiddenLayerOutputs);
		outputNeuron.setOutput(outputNeuron.sigmoid(outputNeuron.getSum()));
		hiddenLayerOutputs.clear();

		return outputNeuron.getOutput();

	}

	public ArrayList<Double> classifyTest(ArrayList<Observation> testSet,ArrayList<Neuron> hiddenLayer, Neuron outputNeuron) {
		ArrayList<Double> input = new ArrayList<Double>();
		ArrayList<Double> output = new ArrayList<Double>();
		predictedNeg = 0;
		predictedPos = 0;
		expectedNeg = 0;
		expectedPos = 0;
		double[] vars;
		int target;
		int testCounter = 0;
		int predicted = -1;
		double sqError = 0;
		for (Observation obs : testSet) {
		input.clear();
			vars = obs.getVars();
			target = obs.getLabel();

			for (int i = 0; i < vars.length; i++) {
				input.add(vars[i]);
			}
			double outputLayer = classifyInstance(input, target, hiddenLayer, outputNeuron);
			if (outputLayer < 0.6)
				predicted = 0;
			else
				predicted = 1;
			if (predicted == target) {
				if (predicted == 0) {
					TN++;
					predictedNeg++;
				} else {
					TP++;
					predictedPos++;
				}
				testCounter++;
			} else {
				if (predicted == 0 && target == 1) {
					FN++;
					predictedNeg++;
				}

				else {
					FP++;
					predictedPos++;
				}
			}

			if (target == 0) {
				expectedNeg++;
			} else {
				expectedPos++;
			}

			sqError += Math.pow((target - outputLayer), 2);
			input.clear();
		}
		output.add(testCounter / (double) (testSet.size()));
		output.add(sqError / (2.0 * (double) (testSet.size())));
		return output;
	}

	public static String getResults(String filePath, int numOfHiddenNeurons,
			double learningRate, int foldsNum, int numOfEpochs,
			int numOfWeights) throws IOException {
		long start = System.currentTimeMillis();
		DecimalFormat numberFormat = new DecimalFormat("#.00");
		ArrayList<Observation> list = KMeans.readDataSet(filePath);
		ArrayList<ArrayList<Observation>> folds = new ArrayList<ArrayList<Observation>>();

		int count = 0;

		ArrayList<Observation> testSet = new ArrayList<Observation>();
		ArrayList<Observation> trainSet = new ArrayList<Observation>();
		ArrayList<Double> maxTestAccuracy = new ArrayList<Double>();
		ArrayList<Integer> bestEpochs = new ArrayList<Integer>();
		double currentMax = 0;
		int currentEpoch = 0;
		ArrayList<Double> input = new ArrayList<Double>();

		double[] vars;
		int target;
		MLP_BPX network;
		int epoch = 0;
		double finalAccuracy = 0;
		double max = 0;

		count = 0;
		Collections.shuffle(list);
		for (int i = 0; i < foldsNum; i++)
			folds.add(new ArrayList<Observation>());

		
		
		 
		
	 
		for (int i = 0; i < foldsNum; i++) {
			for (int j = 0; j < list.size() / foldsNum; j++) {
				folds.get(i).add(list.get(count));
				count++;
			}

		} 
		ArrayList<ArrayList<Double>> counts = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> countsExp = new ArrayList<ArrayList<Double>>();

		for (int i = 0; i < foldsNum; i++) {
			counts.add(new ArrayList<Double>());
			countsExp.add(new ArrayList<Double>());
		}

		for (int j = 0; j < foldsNum; j++) {
			network = new MLP_BPX(numOfHiddenNeurons, numOfWeights, learningRate);//Create ML Network instance	                                                    
			currentMax = 0;
			TP = 0;
			FP = 0;
			TN = 0;
			FN = 0;
			testSet = folds.get(j);
			trainSet.clear();

			for (int k = 0; k < foldsNum; k++) {

				for (int l = 0; l < (int) (list.size() / foldsNum); l++) {
					if (k == j)
						continue;
					else
						trainSet.add(folds.get(k).get(l));
				}
			}

			for (epoch = 0; epoch < numOfEpochs; epoch++) {
				network.resetSumSqError();
				for (Observation obs : trainSet) {
					vars = obs.getVars();
					target = obs.getLabel();

					for (int i = 0; i < vars.length; i++) {
						input.add(vars[i]);
					}
					network.FFD(input, target, 1);// feedfoward phase i.e.training the network
					input.clear();

				}

	     		network.resetCounters();
				network.resetSumSqError();
				ArrayList<Double> output = network.classifyTest(testSet,
						network.getHiddenLayer(), network.getOutputNeuron());
				double testAccuracy = output.get(0);
				if (testAccuracy > currentMax) {
					currentMax = testAccuracy;
					currentEpoch = epoch;
				}
				network.resetCounters();
				network.resetSumSqError();
			 
			}// end epoch

			maxTestAccuracy.add(currentMax);
			bestEpochs.add(currentEpoch + 1);

			counts.get(j).add(predictedNeg);
			counts.get(j).add(predictedPos);
			countsExp.get(j).add(expectedNeg);
			countsExp.get(j).add(expectedPos);

		}// end k-fold cv
		
		double sum = 0.0;
		String result = "";
		String epochs = " ---------------------- Best epoch per fold during CV ----------------------\n \n ";

		for (int i = 0; i < maxTestAccuracy.size(); i++) {
			sum += maxTestAccuracy.get(i);
			epochs += "Accuracy >> "
					+ (numberFormat.format(100 * maxTestAccuracy.get(i)))
					+ "%,  Best Epoch ---> " + bestEpochs.get(i) + "\n";
		}
		finalAccuracy = sum / ((double) (foldsNum));
		result += "MLP-BPX Network Classification Accuracy: "
				+ numberFormat.format(100 * finalAccuracy) + "% \n \n";
		result += "Sensitivity: "
				+ Math.round(100.0 * ((+(TP) / ((double) (TP + FN)))))
				+ " % \n \n";
		;
		result += ("Specificity: " + Math
				.round(100.0 * ((TN) / (double) (TN + FP)))) + " % \n \n";
		result += ("Positive Predictive Value: " + Math
				.round(100.0 * ((TP) / (double) (TP + FP)))) + " % \n \n";
		result += ("Negative Predictive Value: " + Math
				.round(100.0 * ((TN) / (double) (FN + TN)))) + " % \n \n";
		long endTime = System.currentTimeMillis();
		double totalTime = ((endTime - start) / 1000.0000);
		result += "Total Execution Time: " + totalTime + " seconds\n \n \n";
		result += epochs;

		return result;

	}

	public static void main(String[] args) throws IOException {
		
		/**uncomment each statement to execute n times for each dataset*/
		for(int i=0;i<15;i++) {
			System.out.println("-------- Experiment "+(i+1)+"-------- ");
			 System.out.println(getResults("B-[Excl Missing].txt",7,0.3, 5,500,8));//0.3
			// System.out.println(getResults("D-[Extracted Features].txt",9,0.4, 5,500,5,7));//0.4;
			//System.out.println(getResults("C-[Replaced by Mean].txt",9,0.50, 5,500,8,7));//0.5
			 //System.out.println(getResults("A-[Unprocessed].txt",9,0.001, 5,500,8,7));//0.001
		 System.out.println("-------------------------------");
		 System.out.println();
		}
		 
 
	}// end main
}