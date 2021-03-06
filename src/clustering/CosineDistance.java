/**
 * 
 */
package clustering;

import cc.mallet.types.Metric;
import cc.mallet.types.SparseVector;

/**
 * @author hongning
 *
 */
public class CosineDistance implements Metric {

	@Override
	public double distance(SparseVector v1, SparseVector v2) {
		if(v1.twoNorm() == 0 || v2.twoNorm() == 0)
			return 1;
		else
			return 1 - v1.dotProduct(v2)/v1.twoNorm()/v2.twoNorm();
	}

}
