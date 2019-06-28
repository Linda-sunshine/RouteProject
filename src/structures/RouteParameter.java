package structures;

public class RouteParameter {
//	public double m_adaptRatio = 0.8; // The ratio of data for training.
	public String m_dataDir = "./data/balanced_normalize"; // The path for the user data.
	public double m_u = 1.1;// the ratio of the global model. 
	public double m_eta1 = 0.05;
	public double m_eta2 = 0.05;
	public double m_eta3 = 0.05;
	public double m_eta4 = 0.05;

	public boolean m_saveModel = false;
	public boolean m_savePerf = false;
	
	public int m_fvSize = 8;
	public String m_model = "mtlinadapt";

	public double m_sdA = 0.1;
	public double m_sdB = 0.1;
	
	public int m_nuI = 30;

	public int m_fold = 1;
	public int m_perc = 10;

	// Define the parameters used in the algorithm.
	public RouteParameter(String argv[])
	{
		int i;		
		// parse options
		for(i=0;i<argv.length;i++) {
			if(argv[i].charAt(0) != '-') 
				break;
			else if(++i>=argv.length)
				System.exit(1);
			else if (argv[i-1].equals("-dataDir"))
				m_dataDir = argv[i];
			else if (argv[i-1].equals("-model"))
				m_model = argv[i];
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta2"))
				m_eta2 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta3"))
				m_eta3 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-eta4"))
				m_eta4 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-u"))
				m_u = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-fv"))
				m_fvSize = Integer.valueOf(argv[i]);			
			else if (argv[i-1].equals("-saveModel"))
				m_saveModel = Boolean.valueOf(argv[i]);
			else if (argv[i-1].equals("-savePerf"))
				m_savePerf = Boolean.valueOf(argv[i]);
			else if (argv[i-1].equals("-fold"))
				m_fold = Integer.valueOf(argv[i]);
			else if (argv[i-1].equals("-perc")){
				m_perc = Integer.valueOf(argv[i]);
			}
			else
				exit_with_help();
		}
	}
	
	private void exit_with_help()
	{
		System.out.print(
		 "Usage: java execution [options] training_folder\n"
		+"--------------------------------------------------------------------------------\n"
		+"Parameters:\n"
		+"-adaptRatio: the ratio of data for batch training(default 0.8)\n"
		+"-dataDir: the path for the data used for training(default dir \"./data/RouteData/\")\n"
		+"-global: the path for the global svm model weights used in MTLinAdapt\n"
		+"-eta1: coefficient for the regularization (default 0.05)\n"
		+"-eta2: coefficient for the regularization (default 0.05)\n"
		+"-eta3: coefficient for the regularization (default 0.05)\n"
		+"-eta4: coefficient for the regularization (default 0.05)\n"
		+"-u: the ratio of gloal model in each user's personalzied model, w_u = sqrt(u)*w_g + w_u.(dafault 1.1)\n"
		+"-fv: the feature size for the current data set (default 13)\n"
		+"-saveModel: save personalized users models or not (default false)\n"
		+"-savePerf: save each user's performance in one file or not (deafult false)\n"
		+"--------------------------------------------------------------------------------\n"
		);
		System.exit(1);
	}
}
