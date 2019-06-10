package Classifier.supervised;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import structures._Review;
import structures._Review.rType;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class GlobalSVM extends IndividualSVM{

	public GlobalSVM(int classNo, int featureSize) {
		super(classNo, featureSize);
	}
	
	@Override
	public String toString() {
		return String.format("Global-SVM[C:%.3f,bias:%b]", m_C, m_bias);
	}
	
	@Override
	public double train() {
		init();
		int pos = 0, neg = 0;
		
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int trainSize = 0, validUserIndex = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(_AdaptStruct user:m_userList){
			reviews = user.getReviews();		
			boolean validUser = false;
			for(_Review r:reviews) {				
				if (r.getType() == rType.ADAPTATION) {//we will only use the adaptation data for this purpose
					fvs.add(createLibLinearFV(r));
					ys.add(new Double(r.getYLabel()));
					trainSize ++;
					validUser = true;
					// check the class prior
					if(r.getYLabel() == 1) pos++;
					else neg++;
				}
			}
			if (validUser)
				validUserIndex ++;
		}

		System.out.format("[Stat]Among this set of users: pos instances: %d, neg instances: %d\n", pos, neg);
		// Train individual model for each user.
		Problem libProblem = new Problem();
		libProblem.l = trainSize;		
		libProblem.x = new Feature[trainSize][];
		libProblem.y = new double[trainSize];
		for(int i=0; i<trainSize; i++) {
			libProblem.x[i] = fvs.get(i);
		}
		
		if (m_bias) {
			libProblem.n = m_featureSize + 1; // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize;
			libProblem.bias = -1;// no bias term in liblinear.
		}
	
		m_libModel = Linear.train(libProblem, new Parameter(m_solverType, m_C, SVM.EPS));
		for(double w: m_libModel.getWeights())
			System.out.print(w+"\t");
		System.out.println();
		setPersonalizedModel();
		return 0;
	}
	
	@Override
	protected void setPersonalizedModel() {
		for(_AdaptStruct user: m_userList)
			super.setPersonalizedModel(user);
	}
	
	public void saveSupModel(String filename){
		double[] weights = m_libModel.getWeights();
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1;
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			if(m_bias)
				writer.write(sign*weights[m_featureSize]+"\n");
			else
				writer.write(0);
				
			for(int i=0; i<m_featureSize; i++)
					writer.write(weights[i]+"\n");
			
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void savePerf(String perfLocation) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(perfLocation+"/allUsers.perf"));
			for(_AdaptStruct user:m_userList) {
	            StringBuilder buffer = new StringBuilder(512);
	            buffer.append(user.getUserID()+"\t");
//	            for(_Review r: user.getReviews()){
//	            	if(r.getType() == rType.TEST)
//	            }	
	            for(int i=0; i<m_classNo; i++){
	            	for(int j=0; j<m_classNo; j++)
	            		buffer.append(user.getPerfStat().getEntry(i, j)+"\t");
	            }
	            buffer.append("\n");
	            writer.write(buffer.toString());
	        } 
	        writer.close();
		}catch (Exception e) {
			e.printStackTrace();  
		}
	}
}
