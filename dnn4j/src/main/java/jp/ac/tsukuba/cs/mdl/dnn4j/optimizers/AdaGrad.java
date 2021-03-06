package jp.ac.tsukuba.cs.mdl.dnn4j.optimizers;

import com.google.common.collect.Maps;
import jp.ac.tsukuba.cs.mdl.dnn4j.OptimizerArgType;
import jp.ac.tsukuba.cs.mdl.numj.core.NdArray;
import jp.ac.tsukuba.cs.mdl.numj.core.NumJ;

import java.util.Map;

/**
 * Created by yoshihiro on 17/07/11.
 */
public class AdaGrad implements Optimizer {

    private double learningRate;

    private Map<String, NdArray> h;

    public AdaGrad(Map<String, Double> param) {
        learningRate = param.getOrDefault(OptimizerArgType.LEARNING_RATE, 0.01);
    }

    @Override
    public Map<String, NdArray> update(Map<String, NdArray> param, Map<String, NdArray> grad) {
        Map<String, NdArray> result = Maps.newHashMap();
        if (h == null) {
            h = Maps.newHashMap();
            for (Map.Entry<String, NdArray> entry : param.entrySet()) {
                h.put(entry.getKey(), NumJ.zeros(entry.getValue().shape()));
            }
        }

        for (String key : param.keySet()) {

            h.put(
                    key,
                    h.get(key).add(grad.get(key).mul(grad.get(key)))
            );

            result.put(
                    key,
                    param.get(key)
                            .sub(
                                    grad.get(key).div(h.get(key).elementwise(Math::sqrt).add(1e-8)).mul(learningRate)
                            )
            );
        }
        return result;
    }
}
