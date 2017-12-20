package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.OptimizerArgType;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Map;

public class SGD implements Optimizer{

    private double learningRate;

    public SGD(Map<String, Double> params){
        this.learningRate = params.getOrDefault(OptimizerArgType.LEARNING_RATE, 0.01);
    }

    @Override
    public Map<String, NdArray> update(Map<String,NdArray> param, Map<String,NdArray> grad) {
        Map<String, NdArray> result = Maps.newHashMap();
        for (String key: param.keySet()){
            result.put(key, param.get(key).sub(grad.get(key).mul(learningRate)));
        }
        return result;
    }
}
