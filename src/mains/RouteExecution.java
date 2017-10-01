package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import structures.RouteParameter;
import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;

public class RouteExecution {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		RouteParameter param = new RouteParameter(args);
		double trainRatio = 0, adaptRatio = param.m_adaptRatio;
		boolean enforceAdapt = true;

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = param.m_dataDir;
		String globalModel = param.m_globalDir;
		
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setFeatureSize(param.m_fvSize);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.Normalize(3);
		
		RegLR adaptation = null;
		if(param.m_model.equals("mtreglr")){
			adaptation = new MTRegLR(classNumber, param.m_fvSize, null, null);
			((MTRegLR) adaptation).setU(param.m_u);
			((RegLR) adaptation).setR1TradeOff(param.m_eta1);
			adaptation.setLNormFlag(true);

		} else if(param.m_model.equals("mtlinadapt")){
			adaptation = new MTLinAdapt(classNumber, param.m_fvSize, null, 15, globalModel, null, null);
			((MTLinAdapt) adaptation).setParams(param.m_eta1, param.m_eta2, param.m_eta3, param.m_eta4);
			adaptation.setLNormFlag(true);

		} else if(param.m_model.equals("clinadapt")){
			adaptation = new MTCLinAdaptWithDP(classNumber, 15, null, globalModel, null, null);
			
			adaptation.setLNormFlag(false);
			((CLRWithDP) adaptation).setNumberOfIterations(param.m_nuI);
			((CLRWithDP) adaptation).setsdA(param.m_sdA);
			((CLinAdaptWithDP) adaptation).setsdB(param.m_sdB);
			((LinAdapt) adaptation).setR1TradeOffs(param.m_eta1, param.m_eta2);
			((CoLinAdapt) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
		} else{
			System.out.println("The model is not developed...");
		}
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.train();
		adaptation.test();
		
		if(param.m_saveModel)
			adaptation.saveModel("./data/models/"+param.m_model);
		if(param.m_savePerf)
			adaptation.savePerf("./data/"+param.m_model);
	}
}
