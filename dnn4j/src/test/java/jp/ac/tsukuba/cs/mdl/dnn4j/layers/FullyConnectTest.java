package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class FullyConnectTest {
    NdArray weight;
    NdArray bias;
    FullyConnect fullyConnect;

    @Before
    public void init() {
        weight = NumJ.create(new double[]{-1, -2, -3,
                        -4, -5, -6,
                        -7, -8, -9},
                3, 3);
        bias = NumJ.create(new double[]{0.5, 0.6, 0.7}, 1, 3);
        fullyConnect = new FullyConnect(weight, bias);
    }

//    @Test
    public void forward() throws Exception {
        NdArray input = NumJ.create(new double[]{1., 2., 3.}, 1, 3);
        NdArray result = NumJ.create(new double[]{-29.5, -35.4, -41.3}, 1, 3);
        assertTrue(result.sub(fullyConnect.forward(input)).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(new int[]{1, 3}, fullyConnect.forward(input).shape());
    }

//    @Test
    public void backward() throws Exception {
        forward();
        NdArray dout = NumJ.ones(1, 3);
        NdArray dx = NumJ.create(new double[]{-6., -15., -24}, 1, 3);
        assertTrue(dx.sub(fullyConnect.backward(dout)).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(new int[]{1, 3}, fullyConnect.backward(dout).shape());
        NdArray dW = NumJ.create(new double[]{1., 1., 1., 2., 2., 2., 3., 3., 3.,}, 3, 3);
        assertTrue(fullyConnect.getWeightGrad().sub(dW).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(dW.shape(), fullyConnect.getWeightGrad().shape());
        assertTrue(fullyConnect.getBiasGrad().sub(dout).elementwise(Math::abs).sum() < 1e-10);
        assertArrayEquals(new int[]{1, 3}, fullyConnect.getBiasGrad().shape());
    }

}
