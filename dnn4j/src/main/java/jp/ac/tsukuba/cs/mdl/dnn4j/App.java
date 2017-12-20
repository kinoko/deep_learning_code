package jp.ac.tsukuba.cs.mdl.dnn4j;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.Cifar10Dataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.Dataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.dataset.MnistDataset;
import jp.ac.tsukuba.cs.mdl.dnn4j.layers.SigmoidWithLoss;
import jp.ac.tsukuba.cs.mdl.dnn4j.networks.Net;
import jp.ac.tsukuba.cs.mdl.dnn4j.networks.NeuralNet;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class App {

    public static void main(String[] args){

        /*
        データの準備
         */
        Dataset dataset = new MnistDataset();
        int[] inputShape = new int[]{
                dataset.getChannelSize(), dataset.getHeight(), dataset.getWidth()
        };
        NdArray xTrain = dataset.readTrainFeatures().reshape(
                dataset.getTrainSize(),
                dataset.getChannelSize(),
                dataset.getHeight(),
                dataset.getWidth()
        );
        NdArray tTrain = dataset.readTrainLabels();
        NdArray xTest = dataset.readTestFeatures().reshape(
                dataset.getTestSize(),
                dataset.getChannelSize(),
                dataset.getHeight(),
                dataset.getWidth()
        );
        NdArray tTest = dataset.readTestLabels();

        /*
        ネットワークアーキテクチャの設計
         */
        List<Map<String, Integer>> netArgList = constructNetArch();

        /*
        最適化アルゴリズムのパラメータの設定
         */
        Map<String, Double> optimizerParams = Maps.newHashMap();

        /*
        ネットワークの構成
         */
        Net net = new NeuralNet(
                inputShape,
                netArgList,
                new SigmoidWithLoss(),
                0.005 // weight decay lambda
        );

        /*
        訓練を行う
         */
        Trainer trainer = new TrainerImpl(
                net, // network
                xTrain, // input train data
                tTrain, // target train data
                xTest, // input test data
                tTest, // target test data
                100, // epoch num
                100, // mini batch size
                OptimizerType.ADAM, // optimizer
                optimizerParams, // optimizer parameter
                100, // evaluate batch size
                true // verbose
        );
        trainer.train();
    }

    private static List<Map<String, Integer>> constructNetArch() {
        List<Map<String, Integer>> netArgList = new ArrayList<>();

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.CONVOLUTION)
                        .put(NetArgType.FILTER_NUM, 64)
                        .put(NetArgType.FILTER_SIZE, 3)
                        .put(NetArgType.STRIDE, 1)
                        .put(NetArgType.PADDING, 1)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.RELU)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.POOLING)
                        .put(NetArgType.FILTER_SIZE, 2)
                        .put(NetArgType.STRIDE, 2)
                        .put(NetArgType.PADDING, 0)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FLATTEN)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 512)
                        .build()
        );

        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.SIGMOID)
                        .build()
        );


        netArgList.add(
                new MapBuilder<String, Integer>()
                        .put(NetArgType.LAYER_TYPE, LayerType.FULLY_CONNECT)
                        .put(NetArgType.UNIT_NUM, 10)
                        .build()
        );

        return netArgList;
    }
}
