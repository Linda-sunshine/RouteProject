package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import structures.RouteParameter;
import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;

public class RouteExecution {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		RouteParameter param = new RouteParameter(args);
		double trainRatio = 0, adaptRatio = param.m_adaptRatio;
		boolean enforceAdapt = true;
		int featureSize = 13;

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = param.m_dataDir;
				
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.loadTrainIndexes("./data/Indexes.txt");
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.Normalize(2);
		
		MTRegLR adaptation = new MTRegLR(classNumber, featureSize, null, null);
		adaptation.setLNormFlag(false);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setTradeOffParam(param.m_u);
		adaptation.setR1TradeOff(param.m_eta1);
		adaptation.train();
		adaptation.test();
		adaptation.saveModel("./data/RouteModels_mtreglr/");
		adaptation.savePerf("./data/");
	}
}
