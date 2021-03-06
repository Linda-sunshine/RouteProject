package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._SparseFeature;
import structures._thetaStar;
import utils.Utils;
/***
 * Linear transformation matrix with DP.
 * @author lin
 */
public class MTCLinAdaptWithDP extends CLinAdaptWithDP {
	
	protected int m_dimSup;
	protected int[] m_featureGroupMap4SupUsr; // bias term is at position 0
	protected double[] m_supModel; // linear transformation for super user
	
	protected double m_eta3 = 1.0, m_eta4 = 1.0; // will be used to scale regularization term
	
	public MTCLinAdaptWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap, String featureGroup4Sup){
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		loadFeatureGroupMap4SupUsr(featureGroup4Sup);
				
		m_supModel = new double[m_dimSup*2]; // globally shared transformation matrix.
		//construct the new global model for simplicity
		m_supWeights = new double[m_featureSize+1];
	}
	
	public void setR2TradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2 + m_dimSup*2;// we have global here.
	}
	
	//Load global model from file.
	public void loadGlobalModel(HashMap<String, Integer> featureMap, String filename){
			
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line = reader.readLine();
			String features[];
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			
			features = line.split(",");
			System.out.println(features.length + " features are loaded for global model.");
			for(int i=0; i<features.length; i++){
				m_gWeights[i] = Double.valueOf(features[i]);
			}
				
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	@Override
	protected void accumulateClusterModels(){
		super.accumulateClusterModels();

		// we put the global part in the end.
		System.arraycopy(m_supModel, 0, m_models, m_dim*2*m_kBar, m_dimSup*2);
	}
	
	@Override
	protected void initPriorG0() {
		super.initPriorG0();
		
		//sample the global model adaptation parameters
		m_G0.sampling(m_supModel);
	}
	
	@Override
	// R1 over each cluster, R1 over super cluster.
	protected double calculateR1(){
		double R1 = super.calculateR1();
				
		R1 += m_G0.logLikelihood(m_supModel, m_eta3, m_eta4);
		// R1 by super model.
		int offset = m_dim*2*m_kBar;
		for(int k=0; k<m_dimSup; k++){
			m_g[offset+k] += m_eta3 * (m_supModel[k]-m_abNuB[0])/m_abNuB[1]/m_abNuB[1]; // scaling
			m_g[offset+k+m_dimSup] += m_eta4 * (m_supModel[m_dimSup+k]-m_abNuA[0])/m_abNuA[1]/m_abNuA[1];
		}
		
		return R1;
	}
	
	protected double getSupWeights(int n){
		int gid = m_featureGroupMap4SupUsr[n];
		return m_supModel[gid]*m_gWeights[n] + m_supModel[gid+m_dimSup];		
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n, k, s; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		int offset = m_dim*2*cIndex, offsetSup = m_dim*2*m_kBar;
		
		double delta = (review.getYLabel() - logit(review.getSparse(), user)) * weight;
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		// Bias term for individual user.
		g[offset] -= delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		g[offsetSup] -= delta*user.getScaling(0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		g[offsetSup + m_dimSup] -= delta*user.getScaling(0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			g[offsetSup + s] -= delta*user.getScaling(k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			g[offsetSup + m_dimSup + s] -= delta*user.getScaling(k)*fv.getValue(); // a_i*x_di
		}
	}
	
	@Override
	protected double gradientTest() {
		double magC = 0, magS = 0 ;
		int offset = m_dim*2*m_kBar;
		for(int i=0; i<offset; i++)
			magC += m_g[i]*m_g[i];
		for(int i=offset; i<m_g.length; i++)
			magS += m_g[i]*m_g[i];
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for clusters: %.5f, super model: %.5f\n", magC/m_kBar, magS);
		return 0;
	}
	
	// Feature group map for the super user.
	protected void loadFeatureGroupMap4SupUsr(String filename){
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimSup = m_featureSize + 1;
			m_featureGroupMap4SupUsr = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMap4SupUsr[i] = i;
			return;
		} else{// If there is feature grouping for the super user, load it.
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
				String[] features = reader.readLine().split(",");//Group information of each feature.
				reader.close();
				
				m_featureGroupMap4SupUsr = new int[features.length + 1]; //One more term for bias, bias->0.
				m_dimSup = 0;
				//Group index starts from 0, so add 1 for it.
				for(int i=0; i<features.length; i++) {
					m_featureGroupMap4SupUsr[i+1] = Integer.valueOf(features[i]) + 1;
					if (m_dimSup < m_featureGroupMap4SupUsr[i+1])
						m_dimSup = m_featureGroupMap4SupUsr[i+1];
				}
				m_dimSup ++;
			} catch(IOException e){
				System.err.format("[Error]Fail to open super user group file %s.\n", filename);
			}
		}
		
		System.out.format("[Info]Feature group size for super user %d\n", m_dimSup);
	}
	
	// Logit function is different from the father class.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int k, n;
		double[] Au = ((_DPAdaptStruct)u).getThetaStar().getModel();
		double value = Au[0]*getSupWeights(0) + Au[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (Au[k]*getSupWeights(n) + Au[m_dim+k]) * fv.getValue();
		}
		return Utils.logistic(value);
	}
	
	@Override
	protected void setPersonalizedModel() {
		double[] As;
		int ki, ks;
		_DPAdaptStruct user;

		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			As = user.getThetaStar().getModel();
			m_pWeights = new double[m_gWeights.length];
			for(int n=0; n<=m_featureSize; n++){
				ki = m_featureGroupMap[n];
				ks = m_featureGroupMap4SupUsr[n];
				m_pWeights[n] = As[ki]*(m_supModel[ks]*m_gWeights[n] + m_supModel[ks+m_dimSup])+As[ki+m_dim];
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
	
	// Assign the optimized models to the clusters.
	@Override
	protected void setThetaStars(){
		super.setThetaStars();
		
		// Assign model to super user.
		System.arraycopy(m_models, m_dim*2*m_kBar, m_supModel, 0, m_dimSup*2);
	}
	
	@Override
	public String toString() {
		return String.format("CLinAdaptWithDP[dim:%d,supDim:%d,M:%d,alpha:%.4f,#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]", m_dim, m_dimSup, m_M, m_alpha, m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	
	//apply current model in the assigned clusters to users
	@Override
	protected void evaluateModel() {
		for(int i=0; i<m_featureSize+1; i++)
			m_supWeights[i] = getSupWeights(i);
		
		super.evaluateModel();	
	}
	
	public void printClusterInfo(){
		for(int i=0;i<m_userList.size(); i++){
			_DPAdaptStruct user = (_DPAdaptStruct) m_userList.get(i);
			System.out.println(i + "," + user.getThetaStar().getIndex());
		}
	}

	@Override
	public void saveModel(String modelLocation) {
		setPersonalizedModel();
		for(_AdaptStruct user:m_userList) {
			try {
	            BufferedWriter writer = new BufferedWriter(new FileWriter(modelLocation+"/"+user.getUserID()+".classifer"));
	            StringBuilder buffer = new StringBuilder(512);
	            double[] pWeights = user.getPWeights();
	            for(int i=0; i<pWeights.length; i++) {
	            	buffer.append(pWeights[i]);
	            	if (i<pWeights.length-1)
	            		buffer.append(',');
	            }
	            writer.write(buffer.toString());
	            writer.close();
	        } catch (Exception e) {
	            e.printStackTrace(); 
	        } 
		}
		System.out.format("[Info]Save personalized models to %s.", modelLocation);
	}
}
