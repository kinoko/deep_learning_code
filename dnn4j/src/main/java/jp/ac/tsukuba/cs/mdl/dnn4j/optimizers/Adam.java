package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.OptimizerArgType;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Map;

public class Adam implements Optimizer {

    private double learningRate;
    private double beta1;
    private double beta2;
    private int iter = 0;
    private Map<String, NdArray> m;
    private Map<String, NdArray> v;

    public Adam(Map<String, Double> param) {
        this.learningRate = param.getOrDefault(OptimizerArgType.LEARNING_RATE, 0.001);
        this.beta1 = param.getOrDefault(OptimizerArgType.BETA1, 0.9);
        this.beta2 = param.getOrDefault(OptimizerArgType.BETA2, 0.999);
    }

    @Override
    public Map<String, NdArray> update(Map<String, NdArray> param, Map<String, NdArray> grad) {
        Map<String, NdArray> result = Maps.newConcurrentMap();
        if (m == null) {
            m = Maps.newHashMap();
            v = Maps.newHashMap();
            for (Map.Entry<String, NdArray> entry : param.entrySet()) {
                m.put(entry.getKey(), NumJ.zeros(entry.getValue().shape()));
                v.put(entry.getKey(), NumJ.zeros(entry.getValue().shape()));
            }
        }
        iter++;
        double learningRateInT = learningRate * Math.sqrt(1 - Math.pow(beta2, iter)) / (1 - Math.pow(beta1, iter));

        param.keySet().stream().parallel().forEach(key -> {
                    m.put(key, m.get(key).mul(beta1).add(grad.get(key).mul(1 - beta1)));
                    v.put(key, v.get(key).mul(beta2).add(grad.get(key).elementwise(i -> i * i).mul(1 - beta2)));

                    result.put(
                            key,
                            param.get(key).sub(
                                    m.get(key).div(v.get(key).elementwise(Math::sqrt).add(1e-8)).mul(learningRateInT)
                            )
                    );
                }
        );

        for (String key : param.keySet()) {

        }
        return result;
    }
}
