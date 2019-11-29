package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;

import java.io.*;
import java.util.ArrayList;

public class MultiThreadedBinaryRouteAnalyzer extends BinaryRouteAnalyzer {

    protected int m_numberOfCores;
    protected Object m_userLock=null;
    private Object m_corpusLock=null;

    public MultiThreadedBinaryRouteAnalyzer(String tokenModel, int classNo,
                               String providedCV, int Ngram, int threshold, int numberOfCores)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold);
        m_numberOfCores = numberOfCores;
        m_userLock = new Object();
        m_corpusLock = new Object();
    }

    @Override
    public void loadUserDir(String folder){
        if(folder == null || folder.isEmpty())
            return;
        File dir = new File(folder);
        final File[] files=dir.listFiles();
        ArrayList<Thread> threads = new ArrayList<Thread>();
        for(int i=0;i<m_numberOfCores;++i){
            threads.add(  (new Thread() {
                int core;
                @Override
                public void run() {
                    try {
                        for (int j = 0; j + core <files.length; j += m_numberOfCores) {
                            File f = files[j+core];
                            if(f.isFile()){//load the user
                                loadUser(f.getAbsolutePath(), core);
                            }
                        }
                    } catch(Exception ex) {
                        ex.printStackTrace();
                    }
                }

                private Thread initialize(int core ) {
                    this.core = core;
                    return this;
                }
            }).initialize(i));

            threads.get(i).start();
        }
        for(int i=0;i<m_numberOfCores;++i){
            try {
                threads.get(i).join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // process sub-directories
        int count=0;
        for(File f:files )
            if (f.isDirectory())
                loadUserDir(f.getAbsolutePath());
            else
                count++;

        System.out.format("%d users are loaded from %s...\n", count, folder);
    }

    // Load one train/test file for each user here.
    private void loadUser(String filename, int core){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            String userID = extractUserID(file.getName()); //UserId is contained in the filename.

            int yLabel;
            String[] strs;
            _Review review;
            // load the train reviews
            ArrayList<_Review> reviews = new ArrayList<_Review>();
            while((line = reader.readLine()) != null){
                strs = line.split(",");
                if(strs.length == m_featureSize+1){
                    // Construct the new review.
                    yLabel =  Double.valueOf(strs[m_featureSize]).intValue();
                    review = new _Review(m_corpus.getCollection().size(), line, yLabel);
                    review.setType(_Doc.rType.ADAPTATION);
                    AnalyzeDoc(review);
                    reviews.add(review);
                    synchronized (m_corpusLock) {
                        m_corpus.addDoc(review);
                        m_classMemberNo[yLabel]++;
                    }
                }
            }
            reader.close();

            // load the test reviews
            String[] pathStrs = filename.split("/");
            int len = pathStrs[pathStrs.length-1].length() + pathStrs[pathStrs.length-2].length() + 2;
            String testFilename = String.format("%s/Test/%sTest.txt", filename.substring(0, filename.length()-len), userID);
            file = new File(testFilename);
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            while((line = reader.readLine()) != null){
                strs = line.split(",");
                if(strs.length == m_featureSize+1){
                    // Construct the new review.
                    yLabel =  Double.valueOf(strs[m_featureSize]).intValue();
                    review = new _Review(m_corpus.getCollection().size(), line, yLabel);
                    review.setType(_Doc.rType.TEST);
                    AnalyzeDoc(review);
                    reviews.add(review);
                    synchronized (m_corpusLock) {
                        m_corpus.addDoc(review);
                        m_classMemberNo[yLabel]++;
                    }
                }
            }
            reader.close();
            _User user = new _User(userID, m_classNo, reviews);
            synchronized (m_userLock) {
                m_users.add(user); //create new user from the file.
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }


}
