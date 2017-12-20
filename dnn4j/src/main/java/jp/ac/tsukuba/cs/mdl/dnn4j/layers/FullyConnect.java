package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

public class FullyConnect implements Layer {

    private NdArray weight;

    private NdArray bias;

    private NdArray input;

    private NdArray weightGrad;

    private NdArray biasGrad;

    public FullyConnect(NdArray weight, NdArray bias) {
        this.weight = weight;
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

    public void setWeight(NdArray weight) {
        this.weight = weight;
    }

    public void setBias(NdArray bias) {
        this.bias = bias;
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
