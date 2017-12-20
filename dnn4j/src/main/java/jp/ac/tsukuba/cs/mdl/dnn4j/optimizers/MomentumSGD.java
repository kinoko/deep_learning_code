package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.OptimizerArgType;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Map;

/**
 * Created by yoshihiro on 17/07/11.
 */
public class MomentumSGD implements Optimizer{

    private double learningRate;

    private double momentum;

    private Map<String, NdArray> v;

    public MomentumSGD(Map<String, Double> param) {
        learningRate = param.getOrDefault(OptimizerArgType.LEARNING_RATE, 0.01);
        momentum = param.getOrDefault(OptimizerArgType.MOMENTUM, 0.9);
    }

    @Override
    public Map<String, NdArray> update(Map<String, NdArray> param, Map<String, NdArray> grad) {
        Map<String, NdArray> result = Maps.newHashMap();
        if (v == null) {
            v = Maps.newHashMap();
            for (Map.Entry<String, NdArray> entry : param.entrySet()) {
                v.put(entry.getKey(), NumJ.zeros(entry.getValue().shape()));
            }
        }

        for (String key : param.keySet()) {

            v.put(
                    key,
                    v.get(key).mul(momentum).sub(grad.get(key).mul(learningRate))
            );

            result.put(
                    key,
                    param.get(key).add(v.get(key))
            );
        }
        return result;
    }
}
