package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class SigmoidTest {
    public NdArray data;
    public NdArray grad;

    @Before
    public void initalize() {
        data = NumJ.create(new double[]{0.5, 0.5,
                        0.5, 0.5},
                2, 2);

        grad = NumJ.create(new double[]{0.25, 0.25,
                        0.25, 0.25},
                2, 2);
    }

//    @Test
    public void forward() throws Exception {
        Layer layer = new Sigmoid();
        assertTrue(data.sub(layer.forward(NumJ.zeros(2, 2))).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(new int[]{2, 2}, layer.forward(NumJ.zeros(2, 2)).shape());
    }

//    @Test
    public void backward() throws Exception {
        Layer layer = new Sigmoid();
        layer.forward(NumJ.zeros(2, 2));
        assertTrue(grad.sub(layer.backward(NumJ.ones(2, 2))).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(new int[]{2, 2}, layer.backward(NumJ.ones(2, 2)).shape());
    }

}
