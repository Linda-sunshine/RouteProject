package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.TokenizeResult;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._Doc.rType;
import utils.Utils;

public class BinaryRouteAnalyzer extends UserAnalyzer {

	public BinaryRouteAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
	}
	
//	@Override
//	public void loadUser(String filename){
//		try {
//			File file = new File(filename);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
//			String line;
//			String userID = extractUserID(file.getName()); //UserId is contained in the filename.
//			// Skip the first line since it is not instances.
////			reader.readLine();
//
//			int ylabel;
//			String[] strs;
//			_Review review;
//			ArrayList<_Review> reviews = new ArrayList<_Review>();
//			while((line = reader.readLine()) != null){
//				strs = line.split(",");
//				if(strs.length == m_featureSize+1){
//					// Construct the new review.
//					ylabel =  Double.valueOf(strs[m_featureSize]).intValue();
//					review = new _Review(m_corpus.getCollection().size(), line, ylabel);
//					AnalyzeDoc(review);
//					reviews.add(review);
//					m_corpus.addDoc(review);
//					m_classMemberNo[ylabel]++;
//				}
//			}
//			allocateReviews(reviews);
//			m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
//			reader.close();
//		} catch(IOException e){
//			e.printStackTrace();
//		}
//	}


	@Override
	public void loadUser(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.

			int ylabel;
			String[] strs;
			_Review review;
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			while((line = reader.readLine()) != null){
				strs = line.split(",");
				if(strs.length == m_featureSize+1){
					// Construct the new review.
					ylabel =  Double.valueOf(strs[m_featureSize]).intValue();
					review = new _Review(m_corpus.getCollection().size(), line, ylabel);
					review.setType(rType.ADAPTATION);
					AnalyzeDoc(review);
					reviews.add(review);
					m_corpus.addDoc(review);
					m_classMemberNo[ylabel]++;
				}
			}
			_User user = new _User(userID, m_classNo, reviews);
			m_users.add(user); //create new user from the file.
			reader.close();

			String[] pathStrs = filename.split("/");
			int len = pathStrs[pathStrs.length-1].length() + pathStrs[pathStrs.length-2].length() + 2;
			String testFilename = String.format("%s/Test/%sTest.txt", filename.substring(0, filename.length()-len), userID);
			file = new File(testFilename);
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			while((line = reader.readLine()) != null){
				strs = line.split(",");
				if(strs.length == m_featureSize+1){
					// Construct the new review.
					ylabel =  Double.valueOf(strs[m_featureSize]).intValue();
					review = new _Review(m_corpus.getCollection().size(), line, ylabel);
					review.setType(rType.TEST);
					AnalyzeDoc(review);
					reviews.add(review);
					m_corpus.addDoc(review);
					m_classMemberNo[ylabel]++;
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	/*Analyze a document and add the analyzed document back to corpus.*/
	@Override
	protected boolean AnalyzeDoc(_Doc review) {
		String[] strs = review.getSource().split(",");
		_SparseFeature[] fvs = new _SparseFeature[strs.length-1];

		for(int i=0; i<strs.length-1; i++)
			fvs[i] = new _SparseFeature(i, Double.valueOf(strs[i]));
		review.setSpVct(fvs);
		return true;
	}
	
	protected int m_featureSize = 0;
	public void setFeatureSize(int fs){
		m_featureSize = fs;
	}
	public int getRouteFeatureSize(){
		return m_featureSize;
	}
	// Normalize
	public void Normalize(int norm){
		if (norm == 1){
			for(_Doc d: m_corpus.getCollection())			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d: m_corpus.getCollection())			
				Utils.L2Normalization(d.getSparse());
		} else if(norm == 3){//z score globally.
			int ttlObvs = getTotalObvs();
			double[][] obvs = new double[ttlObvs][getRouteFeatureSize()];
			double[] mean = new double[getRouteFeatureSize()];
			double[] var = new double[getRouteFeatureSize()];
			int count = 0;
			//Step 1: collect all reviews.
			for(_User u: m_users){
				for(_Review r: u.getReviews()){
					if(r.getType() == rType.ADAPTATION){
						for(_SparseFeature sf: r.getSparse())
							obvs[count][sf.getIndex()] = sf.getValue();
						count++;
					}
				}
			}
			//Step 2: calculate mean and variance.
			for(int i=0; i<getRouteFeatureSize(); i++){
				mean[i] = calcMean(obvs, i);
				var[i] = calcVar(obvs, i, mean[i]);
			}
			//Step 3: normalize the observations.
			for(_User u: m_users){
				for(_Review r: u.getReviews()){
//					if(r.getType() == rType.ADAPTATION){
						for(_SparseFeature sf: r.getSparse()){
							double val = (sf.getValue()-mean[sf.getIndex()])/var[sf.getIndex()];
							sf.setValue(val);

						}
//					}
				}
			}
		} else if(norm == 4){//z score individually.
			
		}
	}
	
	//Calculate the mean of the i-th column.
	public double calcMean(double[][] arr, int i){
		double sum = 0;
		for(int j=0; j<arr.length; j++)
			sum += arr[j][i];
		return sum/arr.length;
	}
	
	public double calcVar(double[][] arr, int i, double mean){
		double sum = 0;
		for(int j=0; j<arr.length; j++)
			sum += (arr[j][i]-mean)*(arr[j][i]-mean);
		return Math.sqrt(sum/arr.length);
	}
	
	public int getTotalObvs(){
		int count = 0;
		for(_User u: m_users){
			for(_Review r: u.getReviews()){
				if(r.getType() == rType.ADAPTATION) 
					count++;
			}
		}
		return count;
	}
	// Load the indexes for training.
	ArrayList<Integer> m_mostIndexes, m_oneIndexes;
	// Load the instance index for training.
	public void loadTrainIndexes(String filename){
		m_mostIndexes = new ArrayList<Integer>();
		m_oneIndexes = new ArrayList<Integer>();
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;	
			String[] strs;
			// Skip the first line since it is not instances.
			while((line = reader.readLine()) != null){
				strs = line.split("\t");
				if(strs.length == 2)
					m_oneIndexes.add(Integer.valueOf(strs[1])-1);
				m_mostIndexes.add(Integer.valueOf(strs[0])-1);
			} 
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// Previously, we selected some of the reviews for adaptation and some of them for testing.
//	//[0, train) for training purpose
//	//[train, adapt) for adaptation purpose
//	//[adapt, 1] for testing purpose
//	void allocateReviews(ArrayList<_Review> reviews, ArrayList<Integer> indexes) {
//		for(int i=0; i<reviews.size(); i++){
//			if(indexes.contains(i)){
//				reviews.get(i).setType(rType.ADAPTATION);
//				m_adaptSize ++;
//			} else {
//				reviews.get(i).setType(rType.TEST);
//				m_testSize ++;
//			}
//		}
//	}
}
