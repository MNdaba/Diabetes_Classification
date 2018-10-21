package code;
/** NB: cross validation to produce CVE-K-Means is done outside of this class*/


import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
 
 
public class KMeans{
 
	//Number of Clusters. This metric should be related to the number of points
    private int NUM_CLUSTERS;    
    private int NUM_OBS;
    private ArrayList<Observation> observations;
    private ArrayList<Cluster>  clusters;
    
    public KMeans(int numOfClusters, int numOfObs,ArrayList<Observation> observations) {
    	this.observations =  observations;
    	this.clusters = new ArrayList<Cluster>();  
    	this.NUM_CLUSTERS = numOfClusters;
    	this.NUM_OBS = numOfObs;
    }
    
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
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}
    
   
    
    //Initializes the clustering process
     
	public boolean init(int num) {
    	 if(observations.size()>=num*2) {
    	  double [] sums = new double[observations.get(0).getVars().length];
          double [] newCentroidVars = new double[observations.get(0).getVars().length];
          int n_points = observations.size();
           
          for(Observation obs : observations) {
           	for(int i = 0;i <sums.length; i++){
           		sums[i]+=obs.getVars()[i];
           	}
           }
            
           	
           	for(int i = 0;i <newCentroidVars.length; i++){
           		newCentroidVars[i] = sums[i]/n_points;
           	}
           	
           	for(Observation o: observations)
        		o.setD(Observation.distance(o, new Observation(new double[o.getVars().length],0)));//Distance from origin to use as initial sort order
    		Observation.insertionSort(observations);//Sort initial instances by distance from origin
 
    	ArrayList<ArrayList<Observation>> halves = new ArrayList<ArrayList<Observation>>();
    	for(int i = 0; i<num; i++) halves.add(new ArrayList<Observation>());

		int count = 0; 
		if(observations.size()>=num*2){
    	for (int i =0;i<num;i++){
			
    		for (int j=0; j<(int)(observations.size()/num);j++){
    			halves.get(i).add(observations.get(count));
    			count++;
    		}
    	}
		}
		
		/**Clustering*/
		int origSize = observations.size()/num;
		if(observations.size()>=num*2){
    	for(int i = 0; i<halves.size(); i++){
    		Cluster c = new Cluster(i+1);
    		int index = (int)(halves.get(i).size()/2);
    		c.setCentroid(halves.get(i).get(index));
    		halves.get(i).remove(index);
    		clusters.add(c);
 
    	}
    	
		}
		observations.clear();
for(int i = 0; i<halves.size(); i++){
	 for(int j =0; j<origSize-1; j++){
		 observations.add(halves.get(i).get(j));
	 }
	
}
return true;
    	 }//end if
    	 
    	 else {return false;}
 
    }
 
	/**IQR for removing outliers in the dataset*/
	public  double quartile(double[] values, double lowerPercent) {

        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("The data array either is null or does not contain any data.");
        }
        // Rank order the values
        double[] v = new double[values.length];
        System.arraycopy(values, 0, v, 0, values.length);
        Arrays.sort(v);
        int n = (int) Math.round(v.length * lowerPercent / 100);
    return(v[n]);

    }
	
	private void plotClusters() {
    	for (int i = 0; i < NUM_CLUSTERS; i++) {
    		Cluster c = clusters.get(i);
    		c.plotCluster();
    	}
    }
    
	//The process to calculate the K Means, with iterating method.
    public boolean calculate() {
        boolean finish = false;
        int iteration = 0;
        
        // Add in new data, one at a time, recalculating centroids with each new one. 
        while(!finish) {
        	//Clear cluster state
        	clearClusters();      	
        	ArrayList<Observation> lastCentroids = getCentroids();
        	//Assign points to the closer cluster
        	assignCluster();   
            //Calculate new centroids.
        	calculateCentroids(iteration);
        	iteration++;
        	ArrayList<Observation>  currentCentroids = getCentroids();
        	//Calculates total distance between new and old Centroids
        	double distance = 0;
        	for(int i = 0; i < lastCentroids.size(); i++) {
        		distance += Observation.distance(lastCentroids.get(i),currentCentroids.get(i));
        }
           	
        	if(distance == 0) {
        		finish = true;
        	}
        }

        return true;
    }
    
    private void clearClusters() {
    	for(Cluster cluster : clusters) {
    		cluster.clear();
    	}
    }
    
    public ArrayList<Cluster> getClusters(){
    	return this.clusters;
    }
    
    public ArrayList<Observation> getCentroids() {
    	ArrayList<Observation> centroids = new ArrayList<Observation>();
    	for(Cluster cluster : clusters) {
    		Observation aux = cluster.getCentroid();
    		Observation obs = new Observation(aux.getVars(), aux.getLabel());
    		obs.setCluster(aux.getCluster());
    		centroids.add(obs);
    	}
    	return centroids;
    }
    
    public void printCentroids(){
    	ArrayList<Observation>  c = getCentroids();
    	for(Observation obs : c) {
    		obs.printObs();
    	}
    }
    
    public void assignCluster() {
        double max = Double.MAX_VALUE;
        double min = max; 
        int cluster = 0;                 
        double distance = 0.0; 
        
        for(Observation obs : observations) {
        	min = max;
            for(int i = 0; i < NUM_CLUSTERS; i++) {
            	Cluster c = clusters.get(i);
                distance = Observation.distance(obs, c.getCentroid());
                if(distance < min){
                    min = distance;
                    cluster = i;
                }
            }
            obs.setCluster(cluster);
            clusters.get(cluster).addObservation(obs);
        }
    }
    
    /**identify a centroid for each cluster*/
    private void calculateCentroids(int iteration) {
        for(Cluster cluster : clusters) {
           double [] sums = new double[cluster.getCentroid().getVars().length];
           double [] newCentroidVars = new double[cluster.getCentroid().getVars().length];
           ArrayList<Observation> list = cluster.getObservations();
            int n_points = list.size();
            
            for(Observation obs : list) {
            	for(int i = 0;i <sums.length; i++){
            		sums[i]+=obs.getVars()[i];
            	}
            }
            Observation centroid = cluster.getCentroid();
             
            if(n_points > 0) {
            	
            	for(int i = 0;i <newCentroidVars.length; i++){
            		newCentroidVars[i] = sums[i]/n_points;
            	}
            	
            	 if(iteration==0){
                      cluster.addObservation(centroid);
                    }
            	 
            	cluster.setCentroid(new Observation(newCentroidVars,-1));
               
            	
              
            }
        }
    }
}