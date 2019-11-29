package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;
import opennlp.tools.util.InvalidFormatException;
import structures.RouteParameter;
import structures._Doc;
import structures._Review;
import structures._User;

public class MyRouteMain {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 1; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		// ratio for adaptaion
		int featureSize = 10; // They both have 8 features.
        boolean saveModel = false;//"true"
        boolean savePerf = true;//"true"

        int fold = 1;
        String model = "mtlinadapt";//"mtreglr","mtlinadapt", "clinadapt"

        String tokenModel = "./data/Model/en-token.bin"; // Token model.
        String globalModel = String.format("./data/new_global_%d.txt", fold);

        int perc = 20;
        String userFolder = String.format("./data/processeddata/%d/%d", fold, perc);
        BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
        analyzer.setFeatureSize(featureSize);
        analyzer.loadUserDir(userFolder);
        // parameters related with mtreglr
        double u = 0.1;// the ratio of the global model.

        // parameters related with mtclinadapt
        double eta1 = 0.025;
        double eta2 = 0.025;
        double eta3 = 0.075;
        double eta4 = 0.075;

        // parameters related with clinadapt
        double sdA = 0.05;
        double sdB = 0.05;

        int nuI = 30;

        ModelAdaptation adaptation = null;
        if(model.equals("gsvm")){
            adaptation = new GlobalSVM(classNumber, featureSize);
        } else if(model.equals("indsvm")) {
            adaptation = new IndividualSVM(classNumber, featureSize);
        } else if(model.equals("mtreglr")) {
            adaptation = new MTRegLR(classNumber, featureSize, null, null);
            ((MTRegLR) adaptation).setU(u);
            ((RegLR) adaptation).setR1TradeOff(eta1);
            adaptation.setLNormFlag(true);

        } else if (model.equals("mtlinadapt")) {
            adaptation = new MTLinAdapt(classNumber, featureSize, null, 10, globalModel, null, null);
            adaptation.setLNormFlag(true);
            ((LinAdapt) adaptation).setR1TradeOffs(eta1, eta2);
            ((CoLinAdapt) adaptation).setR2TradeOffs(eta3, eta4);
//                adaptation.setPersonalization(false);
        } else if (model.equals("clinadapt")) {
            adaptation = new MTCLinAdaptWithDP(classNumber, featureSize, null, globalModel, null, null);

            adaptation.setLNormFlag(false);
            ((CLRWithDP) adaptation).setNumberOfIterations(nuI);
            ((CLRWithDP) adaptation).setsdA(sdA);
            ((CLinAdaptWithDP) adaptation).setsdB(sdB);

            ((LinAdapt) adaptation).setR1TradeOffs(eta1, eta2);
            ((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(eta3, eta4);
        } else {
            System.out.println("The model is not developed...");
        }
        adaptation.loadUsers(analyzer.getUsers());
        adaptation.setDisplayLv(displayLv);
        adaptation.train();
        adaptation.test();

        if (saveModel)
            adaptation.saveModel("./data/" + model);
        if (savePerf)
            adaptation.savePerf(String.format("./data/output/%s_fold_%d_percentage_%d_acc.txt", model, fold, perc));

    }
}