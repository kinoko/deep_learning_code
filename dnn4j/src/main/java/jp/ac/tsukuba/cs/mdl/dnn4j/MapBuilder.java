package jp.ac.tsukuba.cs.mdl.dnn4j;

import com.google.common.collect.Maps;

import java.util.Map;

public class MapBuilder<K, V> {
    Map<K, V> map;

    public MapBuilder(){
        map = Maps.newHashMap();
    }

    public MapBuilder<K, V> put(K key, V value){
        map.put(key, value);
        return this;
    }

    public Map<K, V> build(){
        return map;
    }
}
