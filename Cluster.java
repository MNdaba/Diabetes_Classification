package code;

import java.util.ArrayList;

 
public class Cluster {
	
	public ArrayList<Observation> Observations;
	public Observation centroid;
	public int id;
	
	//Creates a new Cluster
	public Cluster(int id) {
		this.id = id;
		this.Observations = new ArrayList<Observation>();
		this.centroid = null;
	}
 
	public ArrayList<Observation> getObservations() {
		return Observations;
	}
	
	public void addObservation(Observation obs) {
		Observations.add(obs);
	}
 
	public void setObservations(ArrayList<Observation> Observations) {
		this.Observations = Observations;
	}
 
	public Observation getCentroid() {
		 
		return centroid;
	}
 
	public void setCentroid(Observation centroid) {
		this.centroid = centroid;
	}
 
	public int getId() {
		return id;
	}
	
	public void clear() {
		Observations.clear();
	}
	public void plotCentroid(){
		centroid.printObs();
	}
	public void plotCluster() {
		System.out.println("[Cluster: " + id+"]");
		System.out.print("[Centroid: ");
		centroid.printObs();
		System.out.println();
		System.out.println("[Observations: \n");
		for(Observation p : Observations) {
		p.printObs();
		}
		System.out.println("]");
	}
 
}