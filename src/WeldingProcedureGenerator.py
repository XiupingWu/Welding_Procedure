from collections import OrderedDict
import json;
import joblib
import numpy as np;

class WeldingProcedureGenerator:
    def __init__(self, model_path):
        
        package = joblib.load(model_path)
        self.models = package['models']
        self.material_encoder = package['material_encoder']
        self.feature_names = package['feature_names']
        self.model_type = package.get('model_type', 'dt')  # 添加模型类型信息
        # self.reverse_material = {v:k for k,v in self.material_encoder.items()}
    
    def _build_features(self, inputs):
        try:
            material_code = self.material_encoder[inputs['材质']]
        except KeyError:
            raise ValueError(f"未知材质类型: {inputs['材质']}，支持材质: {list(self.material_encoder.keys())}")
        
        # 基础特征
        features = [
            inputs['厚度'],
            inputs['坡口角度'],
            inputs['钝边'],
            inputs['间隙'],
            inputs['直径'],
            1 if inputs['增透剂'] == '是' else 0,
            material_code,
        ]
        
        # 添加派生特征，与训练时保持一致
        features.extend([
            inputs['厚度'] * inputs['坡口角度'],  # 相互作用特征
            inputs['直径'] / 100.0,  # 归一化直径
            np.log1p(inputs['直径']),  # 对数变换
            inputs['厚度'] ** 2,  # 二次项
            inputs['钝边'] / (inputs['厚度'] + 0.001)  # 钝边比例
        ])
        
        return np.array(features)
    
    def generate(self, inputs):
        # 特征构建
        all_features = self._build_features(inputs)
            
        # 针对每个参数使用特定的特征子集进行预测
        params = {}
        for name, model_info in self.models.items():
            # 提取当前参数对应的特征子集
            feature_indices = model_info['feature_indices']
            selected_features = all_features[feature_indices]
            
            # 使用模型预测
            model = model_info['model']
            params[name] = round(float(model.predict([selected_features])[0]), 1)
        
        # 后处理规则
        params.update(self._post_process(inputs, params))
        return self._format_output(params)
    
    def _post_process(self, inputs, params):
        """后处理补偿规则"""
        processed = {}
        # 峰值比例处理
        # processed['峰值比例%'] = 100 if params['峰值电流'] > 150 else 50
        processed['峰值比例%'] = params['峰值比例%']
        
        # 基值参数计算
        if processed['峰值比例%'] == 100:
            processed.update({'基值电流':0, '基值丝速':0})
        else:
            if params['峰值丝速'] > 500:
                processed['基值丝速'] = round(params['峰值丝速'] * 0.6, 1);
            else:
                processed['基值丝速'] = 0;
            processed['基值电流'] = round(params['峰值电流'] * 0.45, 1)
        
        # 焊接角度计算
        processed['焊接角度'] = self._calculate_angle(inputs)
        return processed
    
    def _calculate_angle(self, inputs):
        base = 360 + inputs['直径']/50;
        # print(self.reverse_material);
        if inputs['材质']== '不锈钢':
            return round(base + inputs['厚度']*0.8, 1)
        return round(base + inputs['厚度']*0.5 - inputs['坡口角度']*0.2, 1)
    
    def _format_output(self, params):
        return OrderedDict([
            ('焊接角度', params['焊接角度']),
            ('峰值电流', params['峰值电流']),
            ('基值电流', params['基值电流']),
            ('峰值丝速', params['峰值丝速']),
            ('基值丝速', params['基值丝速']),
            ('峰值比例%', params['峰值比例%']),
            ('摆动速度', params['摆动速度']),
            ('左侧停留', params['左侧停留']),
            ('焊接速度', params['焊接速度']),
            ('脉冲频率', params['脉冲频率']),
            ('摆动幅度', params['摆动幅度']),
            ('右侧停留', params['右侧停留'])
        ])
    
    def get_model_info(self):
        """获取模型信息，用于调试和分析"""
        model_info = {}
        for param, model_data in self.models.items():
            model_info[param] = {
                '使用特征': model_data['feature_names'],
                '模型类型': self.model_type
            }
        return model_info

# ----------------- 使用示例 -----------------
if __name__ == "__main__":    
    # 生成阶段（服务部署）
    generator = WeldingProcedureGenerator('./trained_models/model_package.pkl')
    
    test_input = {
        "材质": "不锈钢",
        "厚度": 10.0,
        "坡口角度": 30.0,
        "钝边": 1.5,
        "间隙": 0,
        "直径": 133,
        "增透剂": "是",
    }
    
    print("\n生成的焊接工艺:")
    print(json.dumps(generator.generate(test_input), indent=2, ensure_ascii=False))
    
    # print("\n模型信息:")
    # print(json.dumps(generator.get_model_info(), indent=2, ensure_ascii=False))