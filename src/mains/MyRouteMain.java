package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;
import opennlp.tools.util.InvalidFormatException;
import structures.RouteParameter;

public class MyRouteMain {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		// ratio for adaptaion
		double trainRatio = 0, adaptRatio = 0.82;
		boolean enforceAdapt = true;
		int featureSize = 5; // They both have 8 features.
		int dataset = 1;// "2"
		
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
//		String userFolder = String.format("./data/MakeUp/rule_2/Dataset1_homo/",dataset);
		String userFolder = String.format("./data/SyntheticDataset28Systemstart/",dataset);
//		String userFolder = String.format("./data/Dataset%d/format2/",dataset);
//		String globalModel = String.format("./data/gsvm_%d.txt", dataset);
		String globalModel = String.format("./data/gsvm_28.txt", dataset);
		
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setFeatureSize(featureSize);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.Normalize(3);// 3: z score.
		
		boolean saveModel = true;//"true"
		boolean savePerf = true;//"true"
		
		String model = "mtlinadapt";//"mtreglr","mtlinadapt", "clinadapt"
		
		// parameters related with mtreglr
		double u = 0.05;// the ratio of the global model.
		double eta1 = 0.06;

		// parameters related with mtclinadapt
		double eta2 = 0.06;
		double eta3 = 0.02;
		double eta4 = 0.02;

		// parameters related with clinadapt
		double sdA = 0.05;
		double sdB = 0.05;

		int nuI = 30;
		
		RegLR adaptation = null;
		if(model.equals("mtreglr")){
			adaptation = new MTRegLR(classNumber, featureSize, null, null);
			((MTRegLR) adaptation).setU(u);
			((RegLR) adaptation).setR1TradeOff(eta1);
			adaptation.setLNormFlag(true);

		} else if(model.equals("mtlinadapt")){
			adaptation = new MTLinAdapt(classNumber, featureSize, null, 15, globalModel, null, null);
			adaptation.setLNormFlag(true);
			((LinAdapt) adaptation).setR1TradeOffs(eta1, eta2);
			((CoLinAdapt) adaptation).setR2TradeOffs(eta3, eta4);
		} else if(model.equals("clinadapt")){
			adaptation = new MTCLinAdaptWithDP(classNumber, featureSize, null, globalModel, null, null);

			adaptation.setLNormFlag(false);
			((CLRWithDP) adaptation).setNumberOfIterations(nuI);
			((CLRWithDP) adaptation).setsdA(sdA);
			((CLinAdaptWithDP) adaptation).setsdB(sdB);

			((LinAdapt) adaptation).setR1TradeOffs(eta1, eta2);
			((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(eta3, eta4);
		} else{
			System.out.println("The model is not developed...");
		}
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.train();
		adaptation.test();
		
		if(saveModel)
			adaptation.saveModel("./data/"+model);
		if(savePerf)
			adaptation.savePerf("./data/"+model);
		
	}
}