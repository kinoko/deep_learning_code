package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public class ReLU implements Layer {

    private NdArray mask;

    @Override
    public NdArray forward(NdArray input) {
        mask = input.where(x -> x <= 0);
        input.put(mask, 0);
        return input;
    }

    @Override
    public NdArray backward(NdArray dout) {
        dout.put(mask, 0);
        return dout;
    }
}
