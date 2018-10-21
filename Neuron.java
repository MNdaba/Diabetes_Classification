package code;

import java.util.ArrayList;

public class Neuron {
private ArrayList<Double> weights  = new ArrayList<Double> ();
private ArrayList<Double> inputs  = new ArrayList<Double> ();
private double output;
private double summation;
private int layer;
private double error;
private double input;
private Observation centre;
private  ArrayList<Double> deltas = new ArrayList<Double>();
public Neuron(ArrayList<Double> weights){
	for(int i = 0; i<weights.size(); i++){
		this.weights.add(weights.get(i));
		deltas.add(0.0);
	}
	 
}
public void setCentre(Observation obs){
	  this.centre = new Observation(obs.getVars(),obs.getLabel());
}

public Observation getCentre(){
	return this.centre;
}
public void setWeightDelta(ArrayList<Double> deltas){
	this.deltas.clear();
	for(int i = 0; i<deltas.size(); i++){
		this.deltas.add(deltas.get(i));
	}
}
public void setError(double e){
	this.error = e;
}
public void setInputs(ArrayList<Double> inputs){
	this.inputs.clear();
	for(int i = 0; i<inputs.size(); i++){
		this.inputs.add(inputs.get(i));
	 
	}
	 
	
}
public void setWeights(ArrayList<Double> weights){
	this.weights.clear();
	for(int i = 0; i<weights.size(); i++){
		this.weights.add(weights.get(i));
	}
}
	public void setInput(double input){
		this.input = input;
	}
	public void setWeight(int index, Double weight){
		this.weights.set(index, weight);
}
	
	public void setOutput(double out){
		this.output = out;
	}
	
public void setSummation(double sum){
	this.summation = sum;
}
	

public  double sigmoid(double z) {
	return 1.0 / (1.0 + Math.exp(-z));
}


public ArrayList<Double> getWeightDeltas(){
	return this.deltas;
}

public double getInput(){
	return this.input;
}
public ArrayList<Double> getWeights(){
	return this.weights;
}
public double getError(){
	return this.error;
}

public double getOutput(){
	return this.output;
}

public double getSum(){
	double sum = 0.0;
	for(int i=0; i<this.weights.size();i++){
	 	sum = sum+ (this.weights.get(i)*this.inputs.get(i));
	
	}
	 
	sum+=1.0;
	this.summation = sum;
	return sum;
}

public Double getWeigt(int index){
	return weights.get(index);
}

public int getLayer(){
	return this.layer;
}

public int getInputsSize(){
	return this.inputs.size();
}



}
