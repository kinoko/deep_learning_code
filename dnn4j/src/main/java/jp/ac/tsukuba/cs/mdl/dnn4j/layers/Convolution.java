package jp.ac.tsukuba.cs.mdl.dnn4j.layers;

import jp.ac.tsukuba.cs.mdl.dnn4j.Utils;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Arrays;

/**
 * Created by yoshihiro on 17/07/15.
 */
public class Convolution implements Layer {

    private NdArray weight;

    private NdArray bias;

    private int filterHeight;

    private int filterWidth;

    private int stride;

    private int padding;

    private NdArray input;

    private int[] inputShape;

    private int filterNum;

    private NdArray col;

    private NdArray weightGrad;

    private NdArray biasGrad;


    public Convolution(
            NdArray weight,
            NdArray bias,
            int filterNum, int filterHeight, int filterWidth,
            int stride, int padding
    ) {
        this.weight = weight;
        this.bias = bias;
        this.stride = stride;
        this.padding = padding;
        this.filterNum = filterNum;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
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
        inputShape = input.shape();
        this.input = input;
        int[] weightShape = weight.shape();
        int outHeight = Utils.computeOutputSize(inputShape[2], filterHeight, stride, padding);
        int outWidth = Utils.computeOutputSize(inputShape[3], filterWidth, stride, padding);

        col = Utils.im2col(input, filterHeight, filterWidth, stride, padding);

        // ここから変更 重み行列を掛ける
        NdArray out = NumJ.zeros(0);
        // ここまで変更

        out = out.reshape(inputShape[0], outHeight, outWidth, filterNum).transpose(0, 3, 1, 2);

        return out;
    }

    @Override
    public NdArray backward(NdArray dout) {
        int[] weightShape = weight.shape();
        dout = dout.transpose(0, 2, 3, 1).reshape(dout.size() / filterNum, filterNum);

        // ここから実装 重みトバイアス、入力のの勾配を計算し、weightGrad、biasGrad、dcolに出力する
        // このの操作はFCとほぼ同じ
        NdArray dcol = NumJ.zeros(0);
        // ここまで実装

        return Utils.col2im(dcol, inputShape, filterHeight, filterWidth, stride, padding);
    }
}
