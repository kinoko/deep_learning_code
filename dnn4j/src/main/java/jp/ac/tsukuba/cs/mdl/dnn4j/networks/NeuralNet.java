package jp.ac.tsukuba.cs.mdl.dnn4j.networks;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.LayerType;
import jp.ac.tsukuba.cs.mdl.dnn4j.NetArgType;
import jp.ac.tsukuba.cs.mdl.dnn4j.Utils;
import jp.ac.tsukuba.cs.mdl.dnn4j.layers.*;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public class NeuralNet implements Net {

    private static final String LAYER = "Layer";
    private static final String WEIGHT = "Weight";
    private static final String BIAS = "Bias";
    private double weightDecayLambda;

    protected Map<String, Layer> layers = new LinkedHashMap<>();

    protected LastLayer lastLayer;

    /**
     * Parameters
     * ----------
     *
     * @param inputShape        : 入力の形状
     * @param netArgumentList   : ネットワークのパラメータ数（構造）を決定するパラメータの層毎のリスト
     * @param lastLayer         : LastLayer Instance
     * @param weightDecayLambda : Weight Decay（L2ノルム）の強さ
     */

    public NeuralNet(
            int[] inputShape,
            List<Map<String, Integer>> netArgumentList,
            LastLayer lastLayer,
            double weightDecayLambda
    ) {
        this.weightDecayLambda = weightDecayLambda;

        LayerInfo layerInfo = new LayerInfo(inputShape[0], inputShape[1], inputShape[2]);

        for (int i = 0; i < netArgumentList.size(); i++) {
            Map<String, Integer> netArgument = netArgumentList.get(i);
            int nextLayerType = 0;
            if (i + 1 < netArgumentList.size()) {
                nextLayerType = netArgumentList.get(i + 1).get(NetArgType.LAYER_TYPE);
            }
            switch ((netArgument.get(NetArgType.LAYER_TYPE))) {
                case LayerType.CONVOLUTION:
                    layers.put(
                            LAYER + (i + 1),
                            constructConvolutionLayer(
                                    netArgument,
                                    nextLayerType,
                                    layerInfo.inputFilterNum
                            )
                    );
                    layerInfo.update(netArgument);
                    break;

                case LayerType.POOLING:
                    layers.put(
                            LAYER + (i + 1),
                            new Pooling(
                                    netArgument.get(NetArgType.FILTER_SIZE),
                                    netArgument.get(NetArgType.FILTER_SIZE),
                                    netArgument.get(NetArgType.STRIDE),
                                    netArgument.get(NetArgType.PADDING)
                            )
                    );
                    layerInfo.update(netArgument);
                    break;

                case LayerType.FULLY_CONNECT:
                    int inputSize = layerInfo.totalInputSize;
                    FullyConnect fullyConnect = constructFullyConnectLayer(netArgument, nextLayerType, inputSize);

                    layers.put(
                            LAYER + (i + 1),
                            fullyConnect
                    );
                    layerInfo.update(netArgument);
                    break;
                case LayerType.FLATTEN:
                    layers.put(LAYER + (i + 1), new Flatten());
                    break;
                case LayerType.SIGMOID:
                    layers.put(LAYER + (i + 1), new Sigmoid());
                    break;
                case LayerType.RELU:
                    layers.put(LAYER + (i + 1), new ReLU());
                    break;
                default:
                    throw new IllegalArgumentException("未定義のレイヤーを作成しようとしています");
            }
        }

        this.lastLayer = lastLayer;
    }

    @Override
    public NdArray predict(NdArray x) {

        for (Layer layer : layers.values()) {
            x = layer.forward(x);
        }
        return x;
    }

    @Override
    public double accuracy(NdArray x, NdArray t) {
        NdArray y = predict(x).argmax(1);
        if (t.dim() != 1) {
            t = t.argmax(1);
        }
        return t.eq(y).sum() / x.shape()[0];
    }

    @Override
    public double forward(NdArray x, NdArray t) {
        NdArray y = predict(x);

        double weightDecay = 0;

        for (Layer layer : layers.values()) {
            if (layer instanceof Convolution) {
                Convolution conv = (Convolution) layer;
                NdArray w = conv.getWeight();
                weightDecay += 0.5 * weightDecayLambda * w.elementwise(e -> Math.pow(e, 2)).sum();
            } else if (layer instanceof FullyConnect) {
                FullyConnect fullyConnect = (FullyConnect) layer;
                NdArray w = fullyConnect.getWeight();
                weightDecay += 0.5 * weightDecayLambda * w.elementwise(e -> Math.pow(e, 2)).sum();
            }
        }

        return lastLayer.forward(y, t) + weightDecay;
    }

    @Override
    public NdArray backward(double dout) {
        NdArray _dout = lastLayer.backward(dout);
        List<Layer> layerList = Lists.newArrayList(layers.values());
        Collections.reverse(layerList);
        for (Layer layer : layerList) {
            _dout = layer.backward(_dout);
        }
        return _dout;
    }

    @Override
    public Map<String, NdArray> getParameters() {
        Map<String, NdArray> result = Maps.newHashMap();
        for (String key : layers.keySet()) {
            if (layers.get(key) instanceof Convolution) {
                Convolution convolution = (Convolution) layers.get(key);
                result.put(
                        key + WEIGHT,
                        convolution.getWeight()
                );
                result.put(key + BIAS, convolution.getBias());
            } else if (layers.get(key) instanceof FullyConnect) {
                FullyConnect fullyConnect = (FullyConnect) layers.get(key);
                result.put(
                        key + WEIGHT,
                        fullyConnect.getWeight()
                );
                result.put(key + BIAS, fullyConnect.getBias());
            }
        }
        return result;
    }

    @Override
    public void setParameters(Map<String, NdArray> params) {
        for (String key : layers.keySet()) {
            if (layers.get(key) instanceof Convolution) {
                Convolution convolution = (Convolution) layers.get(key);
                convolution.setWeight(params.get(key + WEIGHT));
                convolution.setBias(params.get(key + BIAS));
            } else if (layers.get(key) instanceof FullyConnect) {
                FullyConnect fullyConnect = (FullyConnect) layers.get(key);
                fullyConnect.setWeight(params.get(key + WEIGHT));
                fullyConnect.setBias(params.get(key + BIAS));
            }
        }
    }

    @Override
    public Map<String, NdArray> getGradient() {

        Map<String, NdArray> grads = Maps.newHashMap();
        for (String key : layers.keySet()) {
            if (layers.get(key) instanceof Convolution) {
                Convolution convolution = (Convolution) layers.get(key);
                grads.put(
                        key + WEIGHT,
                        convolution.getWeightGrad().add(convolution.getWeight().mul(weightDecayLambda))
                );
                grads.put(key + BIAS, convolution.getBiasGrad());
            } else if (layers.get(key) instanceof FullyConnect) {
                FullyConnect fullyConnect = (FullyConnect) layers.get(key);
                grads.put(
                        key + WEIGHT,
                        fullyConnect.getWeightGrad().add(fullyConnect.getWeight().mul(weightDecayLambda))
                );
                grads.put(key + BIAS, fullyConnect.getBiasGrad());
            }
        }
        return grads;
    }

    private FullyConnect constructFullyConnectLayer(
            Map<String, Integer> netArgument,
            int nextLayerType,
            int inputSize) {
        double scale;
        switch (nextLayerType) {
            case LayerType.RELU:
                scale = Math.sqrt(2.0 / netArgument.get(NetArgType.UNIT_NUM));
                break;
            case LayerType.SIGMOID:
                scale = Math.sqrt(1.0 / netArgument.get(NetArgType.UNIT_NUM));
                break;
            default:
                scale = 0.01;
        }
        return new FullyConnect(
                NumJ.normal(0, 1, inputSize,
                        netArgument.get(NetArgType.UNIT_NUM)).mul(scale),
                NumJ.zeros(1, netArgument.get(NetArgType.UNIT_NUM))
        );
    }

    private Convolution constructConvolutionLayer(
            Map<String, Integer> netArgument,
            int layerType,
            int inputFilterNum
    ) {
        double scale;
        int filterSize = netArgument.get(NetArgType.FILTER_SIZE);
        int filerNum = netArgument.get(NetArgType.FILTER_NUM);
        switch (layerType) {
            case LayerType.RELU:
                scale = Math.sqrt(
                        2.0 / inputFilterNum / filterSize / filterSize
                );
                break;
            case LayerType.SIGMOID:
                scale = Math.sqrt(
                        1.0 / inputFilterNum / filterSize / filterSize
                );
                break;
            default:
                scale = 0.01;
        }
        return new Convolution(
                NumJ.normal(
                        0, // average
                        1, // std
                        filerNum,
                        inputFilterNum,
                        filterSize,
                        filterSize
                ).mul(scale),
                NumJ.zeros(1, filerNum),
                netArgument.get(NetArgType.STRIDE),
                netArgument.get(NetArgType.PADDING)
        );
    }

    private class LayerInfo {
        private int inputFilterNum;
        private int height;
        private int width;
        private int totalInputSize;

        public LayerInfo(int inputFilterNum, int height, int width) {
            this.inputFilterNum = inputFilterNum;
            this.height = height;
            this.width = width;
            totalInputSize = inputFilterNum * height * width;
        }

        public void update(Map<String, Integer> netArgument) {
            if (netArgument.get(NetArgType.LAYER_TYPE) == LayerType.FULLY_CONNECT) {
                totalInputSize = netArgument.get(NetArgType.UNIT_NUM);
                height = 1;
                width = 1;
                inputFilterNum = totalInputSize;
            } else {
                int filterNum = netArgument.getOrDefault(NetArgType.FILTER_NUM, inputFilterNum);
                int filterSize = netArgument.get(NetArgType.FILTER_SIZE);
                int stride = netArgument.get(NetArgType.STRIDE);
                int padding = netArgument.get(NetArgType.PADDING);
                height = Utils.computeOutputSize(
                        height,
                        filterSize,
                        stride,
                        padding
                );
                width = Utils.computeOutputSize(
                        width,
                        filterSize,
                        stride,
                        padding
                );
                inputFilterNum = filterNum;
                totalInputSize = filterNum * height * width;
            }
        }
    }

}
