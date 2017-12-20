package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArrayImpl;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;
import org.junit.Test;

import static org.junit.Assert.*;

public class ReLUTest {
    @Test
    public void forward() throws Exception {
        Layer relu = new ReLU();
        NdArray input = new NdArrayImpl(new int[]{2, 3}, new double[]{1, -1, 0, 1, -1, 0});
        assertEquals(new NdArrayImpl(new int[]{2, 3}, new double[]{1, 0, 0, 1, 0, 0}), relu.forward(input));
        assertEquals(new NdArrayImpl(new int[]{2, 3}, new double[]{1, 0, 0, 1, 0, 0}), relu.backward(NumJ.ones(2, 3)));
    }

    @Test
    public void backward() throws Exception {
    }

}