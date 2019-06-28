package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
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

		RouteParameter param = new RouteParameter(args);
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = String.format("%s/%d/%d/", param.m_dataDir, param.m_fold, param.m_perc);
		String globalModel = String.format("./data/global_%d.txt", param.m_fold);
		
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setFeatureSize(param.m_fvSize);
		analyzer.loadUserDir(userFolder);
		analyzer.Normalize(3);
		
		ModelAdaptation adaptation = null;
		if(param.m_model.equals("gsvm")){
			adaptation = new GlobalSVM(classNumber, param.m_fvSize);
		} else if(param.m_model.equals("indsvm")){
			adaptation = new IndividualSVM(classNumber, param.m_fvSize);
		} else if(param.m_model.equals("mtreglr")){
			adaptation = new MTRegLR(classNumber, param.m_fvSize, null, null);
			((MTRegLR) adaptation).setU(param.m_u);
			((RegLR) adaptation).setR1TradeOff(param.m_eta1);
			adaptation.setLNormFlag(true);
		} else if(param.m_model.equals("mtlinadapt")){
			adaptation = new MTLinAdapt(classNumber, param.m_fvSize, null, 15, globalModel, null, null);
			((MTLinAdapt) adaptation).setParams(param.m_eta1, param.m_eta2, param.m_eta3, param.m_eta4);
			adaptation.setLNormFlag(true);
		} else if(param.m_model.equals("clinadapt")){
			adaptation = new MTCLinAdaptWithDP(classNumber, param.m_fvSize, null, globalModel, null, null);
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
		adaptation.train();
		adaptation.test();
		
		if(param.m_saveModel)
			adaptation.saveModel(String.format("./data/output/Weights_model_%s_fold_%d_perc_%d.txt", param.m_model, param.m_fold, param.m_perc));
		if(param.m_savePerf)
			adaptation.savePerf(String.format("./data/output/Performance_model_%s_fold_%d_perc_%d.txt", param.m_model, param.m_fold, param.m_perc));
	}
}
