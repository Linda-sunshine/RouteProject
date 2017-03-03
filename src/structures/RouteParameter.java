package structures;

public class RouteParameter {
	public double m_adaptRatio = 0.5; // The ratio of data for training.
	public String m_dataDir = "./data/RouteData/"; // The path for the user data.
	public double m_u = 1.1;// the ratio of the global model. 
	public double m_eta1 = 0;
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
			else if( argv[i-1].equals("-adaptRatio"))
				m_adaptRatio = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-dataDir"))
				m_dataDir = argv[i];
			else if (argv[i-1].equals("-eta1"))
				m_eta1 = Double.valueOf(argv[i]);
			else if (argv[i-1].equals("-u"))
				m_u = Double.valueOf(argv[i]);
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
		+"-u: the ratio of gloal model in each user's personalzied model, w_u = sqrt(u)*w_g + w_u.(dafault 1.1)\n"
		+"-eta1: coefficient for the regularization (default 0)\n"
		+"--------------------------------------------------------------------------------\n"
		);
		System.exit(1);
	}
}
