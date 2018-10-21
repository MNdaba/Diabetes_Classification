package code;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
 
public class Observation  {
 
    private double x = 0;
    private double y = 0;
    private int cluster_number = 0;
    private double[] vars;
    private int label = -1;
    private double distance = Double.MAX_VALUE;
    private int classified = -2;
 private double weight = 0;
 private int predictedLabel = -1;
 
    public Observation(double[] vars, int label)
    {
        this.vars = new double[vars.length];
    	for (int i = 0; i<vars.length; i++){
    		this.vars [i] = vars[i];
    	}
    	this.label = label;
    	this.predictedLabel = label;
    	
    }
    public void setD(double d){
    	this.distance = d;
    }
   public void setWeight(double weight){
	  this.weight = weight; 
   }
   
   public int getPredictedLabel(){
	   return this.predictedLabel;
   }
   
   public void setPredictedLabel(int label){
	   this.predictedLabel = label;
   }
   public int getClassification(){
	   return this.classified;
   }
   public double getWeight(){
	   return this.weight;
   }
    public double getD(){
    	return this.distance;
    }
    
    public void setClassification(int classification){
 	     this.classified = classification;
    }
    
    public void setAttribute(double x, int index) {
    	this.vars[index] = x;
    }
    public void setVar(double[] x ) {
        this.vars = x;
    }
    
    public void setLabel(int label){
    	this.label = label;
    }
    public double [] getVars()  {
        return this.vars;
    }
     
    public int getLabel(){
    	return this.label;
    }
    public void setCluster(int n) {
        this.cluster_number = n;
    }
    
    public int getCluster() {
        return this.cluster_number;
    }
    
    
    public static void insertionSort(ArrayList<Observation> Array) {

        int i,j;

        for (i = 1; i < Array.size(); i++) {
        	double [] vars = new double[Array.get(i).getVars().length];
        	int label = Array.get(i).getLabel();
        	double distance = Array.get(i).getD();
        	for(int k = 0; k<vars.length;k++) vars[k] = Array.get(i).getVars()[k];
            Observation temp = new Observation(vars,label);
            temp.setD(distance); 
            j = i;
            while((j > 0) && (Array.get(j - 1).getD() > temp.getD())) {
                Array.set(j,Array.get(j - 1));
                j--;
            }
            Array.set(j,temp);
        }


    }
    
    //Calculates the distance between two points.
    protected static double distance(Observation p, Observation centroid) {
       double d = 0;
       double d2 = 0;
    	for (int i = 0; i<p.getVars().length-1; i++) d2+=Math.pow((centroid.getVars()[i] - p.getVars()[i]), 2);
    return Math.sqrt(d2);
    }
    
     
    public void printObs(){
    	for (int i = 0; i<vars.length; i++){
    		System.out.print(this.vars [i]+", ");
    		
    	}
    	System.out.println(this.label);
    	System.out.println();
    
    }
	 
     }