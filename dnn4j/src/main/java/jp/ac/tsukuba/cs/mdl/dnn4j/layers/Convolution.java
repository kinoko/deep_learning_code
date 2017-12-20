package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.dnn4j.Utils;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

/**
 * Created by yoshihiro on 17/07/15.
 */
public class Convolution implements Layer {

    private NdArray weight;

    private NdArray bias;

    private int stride;

    private int padding;

    private int[] inputShape;

    private NdArray weightGrad;

    private NdArray biasGrad;

    public Convolution(NdArray weight, NdArray bias, int stride, int padding) {
        this.weight = weight;
        this.bias = bias;
        this.stride = stride;
        this.padding = padding;
    }

    public Convolution(NdArray weight, NdArray bias) {
        this(weight, bias, 1, 0);
    }

    public void setWeight(NdArray weight) {
        this.weight = weight;
    }

    public void setBias(NdArray bias) {
        this.bias = bias;
    }

    public NdArray getWeight() {
        return weight;
    }

    public NdArray getBias() {
        return bias;
    }

    public NdArray getWeightGrad() {
        return weightGrad;
    }

    public NdArray getBiasGrad() {
        return biasGrad;
    }

    @Override
    public NdArray forward(NdArray input) {
        return null;
    }

    @Override
    public NdArray backward(NdArray dout) {
        return null;
    }
}
