package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;

public class MyRouteMain {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		double trainRatio = 0, adaptRatio = 0.8;
		boolean enforceAdapt = true;
		int featureSize = 8; // They both have 8 features.
		int dataset = 1;// "2"
		
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = String.format("./data/Dataset%d/Format2",dataset); 
		String globalModel = String.format("./data/gsvm_%d.txt", dataset);
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setFeatureSize(featureSize);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.Normalize(3);// 3: z score.
//		
//		for(int i=1; i<11; i++){
//			for(int j=1; j<11; j++){
//		double e1 = i*0.001, e2 = j*0.001;
//		MTLinAdapt adaptation = new MTLinAdapt(classNumber, featureSize, null, 15, globalModel, null, null);
//		adaptation.setParams(e1, e1, e2, e2);

//		MTRegLR adaptation = new MTRegLR(classNumber, featureSize, null, null);
//		adaptation.setLNormFlag(true);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setU(1.1);
//		adaptation.setR1TradeOff(0.001);
//		adaptation.train();
//		adaptation.test();
//		}}
//		
		GlobalSVM gsvm = new GlobalSVM(classNumber, featureSize);
		gsvm.loadUsers(analyzer.getUsers());
		gsvm.setC(1);
		gsvm.setBias(true);
		gsvm.train();
		gsvm.test();
//		gsvm.saveSupModel("gsvm_2.txt");
//		gsvm.savePerf("./data/");
//		gsvm.saveModel("./data/gsvm/");
////		for(_User u: analyzer.getUsers())
////			u.getPerfStat().clear();
////		
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, featureSize);
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		IndividualSVM indsvm = new IndividualSVM(classNumber, featureSize);
//		indsvm.setC(0.1);
//		indsvm.setBias(false);
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.train();
//		indsvm.test();
		
	}
}
