package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import utils.Utils;

public class _DPAdaptStruct extends _LinAdaptStruct {

	_thetaStar m_thetaStar = null;
	double[] m_cluPosterior;
	
	public static int[] m_featureGroupMap;
	
	public _DPAdaptStruct(_User user) {
		super(user, 0); // will not perform adaptation
	}
	// added by Lin.
	public _DPAdaptStruct(_User user, String id){
		super(user, 0);
		m_id = Integer.valueOf(id);
		m_adaptStartPos = 0;
		m_adaptEndPos = 10;
		resetAdaptPtr();
	}

	public _DPAdaptStruct(_User user, int dim) {
		super(user, dim);
	}
	
	public _thetaStar getThetaStar(){
		return m_thetaStar;
	}
	
	public void setThetaStar(_thetaStar s){
		m_thetaStar = s;
	}
	
	public void setClusterPosterior(double[] posterior) {
		if (m_cluPosterior==null || m_cluPosterior.length != posterior.length)
			m_cluPosterior = new double[posterior.length];
		System.arraycopy(posterior, 0, m_cluPosterior, 0, posterior.length);
	}
	
	@Override
	public double getScaling(int k){
		return m_thetaStar.getModel()[k];
	}
	
	@Override
	public double getShifting(int k){
		return m_thetaStar.getModel()[m_dim+k];
	}
	
	public double evaluate(_Doc doc) {
		double prob = 0, sum;
		
		if (m_dim==0) {//not adaptation based
			for(int k=0; k<m_cluPosterior.length; k++) {
				sum = Utils.dotProduct(CLRWithDP.m_thetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				prob += m_cluPosterior[k] * Utils.logistic(sum); 
			}			
		} else {
			int n, m;
			double As[];
			for(int k=0; k<m_cluPosterior.length; k++) {
				As = CLRWithDP.m_thetaStars[k].getModel();

				sum = As[0]*CLinAdaptWithDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				
				prob += m_cluPosterior[k] * Utils.logistic(sum); 
			}
		}
		

		//accumulate the prediction results during sampling procedure
		doc.m_pCount ++;
		doc.m_prob += prob; //>0.5?1:0;
		
		return prob;
	}
	
	@Override
	public int predict(_Doc doc) {
		double prob = 0;
		if (doc.m_pCount==0)//this document has not been tested yet??
			prob = evaluate(doc);
		else
			prob = doc.m_prob/doc.m_pCount;
		return prob>=0.5 ? 1:0;
	}
}
