package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by riku on 7/5/17.
 */
public class SGDTest {
    @Test
    public void update() throws Exception {
        Map<String, NdArray> params = new HashMap<String, NdArray>();
        NdArray w = NumJ.create(new double []{1,2,3,4,5,6} ,2, 3);
        params.put("W", w);
        SGD sgd = new SGD(new HashMap<>());

        NdArray grad = w.mul(100.0);
        Map<String, NdArray> grads = new HashMap<String, NdArray>();
        grads.put("W", grad);

        assertEquals(NumJ.zeros(2, 3), sgd.update(params, grads).get("W"));
    }

}