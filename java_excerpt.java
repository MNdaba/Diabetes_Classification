ArrayList<Observation> allData = KMeans.readDataSet(filePath);//Read in dataset
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
				
				for(int i=0;i<3;i++) {/**Add 3 samples out of 10 into test set (creating 30% of test cases)*/
					for(Observation o: data.get(i))
						testSet.add(o);
				}
				
				for(int i=0;i<7;i++) {/**Add 7 samples out of 10 into train set (creating 70% of train cases)*/
					for(Observation o: data.get(i))
						trainingSet.add(o);
				}
				
}
