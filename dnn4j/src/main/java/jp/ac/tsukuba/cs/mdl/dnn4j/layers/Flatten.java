package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public class Flatten implements Layer {

    private int[] inputShape;

    @Override
    public NdArray forward(NdArray input) {
        inputShape = input.shape();
        return input.reshape(inputShape[0], input.size() / inputShape[0]);
    }

    @Override
    public NdArray backward(NdArray dout) {
        return dout.reshape(inputShape);
    }
}
