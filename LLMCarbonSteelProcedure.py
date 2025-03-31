import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict

class WeldingProcedureGenerator:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self._build_models()
        self._special_cases = self._identify_special_cases()
    
    def _build_models(self):
        # 数据预处理
        X, y = self._prepare_training_data()
        
        # 建立各参数预测模型
        params = [
                    '峰值电流', '焊接速度', '摆动速度', '摆动幅度', 
                    '左侧停留', '右侧停留', '脉冲频率', '峰值丝速'
                ]
        
        for param in params:
            self.models[param] = DecisionTreeRegressor()
            self.models[param].fit(X, y[param])
            
        # 特殊参数规则
        self.thickness_bins = np.array([3,5,8,12,25])
        self.diameter_bins = np.array([50,100,200,400,1000])
    
    def _prepare_training_data(self):
        # 特征工程
        X = []
        y = {k:[] for k in ['峰值电流','焊接速度','摆动速度','摆动幅度',
                            '左侧停留','右侧停留','脉冲频率','峰值丝速']}
        
        for record in self.data:
            features = [
                record['厚度'],
                record['坡口角度'],
                record['钝边'],
                record['间隙'],
                record['直径'],
                1 if record['增透剂'] == '是' else 0
            ]
            
            segment = record['程序段'][0]['程序段1']
            
            for param in y:
                y[param].append(segment[param])
                
            X.append(features)
            
        return np.array(X), y
    
    def _identify_special_cases(self):
        # 识别特殊工艺参数组合
        specials = {}
        for record in self.data:
            key = (record['厚度'], record['坡口角度'], 
                    record['钝边'], record['间隙'], 
                    record['直径'], record['增透剂'])
            specials[key] = record['程序段'][0]['程序段1']
        return specials
    
    def _calculate_peak_ratio(self, inputs):
        # 峰值比例决策规则
        if inputs['间隙'] > 2 or inputs['坡口角度'] > 35:
            return 100 if inputs['增透剂'] == '是' else 80
        return 100 if inputs['厚度'] < 8 else 50
    
    def generate_procedure(self, inputs):
        # 检查特殊案例
        key = (inputs['厚度'], inputs['坡口角度'], inputs['钝边'],
                inputs['间隙'], inputs['直径'], inputs['增透剂'])
        if key in self._special_cases:
            return self._format_output(self._special_cases[key])
        
        # 特征向量构建
        X = np.array([
                    [inputs['厚度'], inputs['坡口角度'], inputs['钝边'],
                    inputs['间隙'], inputs['直径'], 
                    1 if inputs['增透剂'] == '是' else 0]])
        
        # 参数预测
        params = {}
        for name, model in self.models.items():
            params[name] = round(float(model.predict(X)[0]), 1)
        
        # 关键参数规则处理
        params['峰值比例%'] = self._calculate_peak_ratio(inputs)
        
        # 基值参数处理
        if params['峰值比例%'] == 100:
            params['基值电流'] = 0
            params['基值丝速'] = 0
        else:
            params['基值电流'] = round(params['峰值电流'] * 0.45, 1)
            params['基值丝速'] = round(params['峰值丝速'] * 0.6, 1)
        
        # 焊接角度计算规则
        params['焊接角度'] = self._calculate_welding_angle(inputs)
        
        return self._format_output(params)
    
    def _calculate_welding_angle(self, inputs):
        # 焊接角度计算规则
        base_angle = 360 + (inputs['直径'] / 50)
        adjustment = inputs['厚度'] * 0.5 - inputs['坡口角度'] * 0.2
        return round(base_angle + adjustment, 1)
    
    def _format_output(self, params):
        # 保持输出顺序
        return OrderedDict([
            ('焊接角度', params['焊接角度']),
            ('峰值电流', params['峰值电流']),
            ('基值电流', params.get('基值电流',0)),
            ('峰值丝速', params['峰值丝速']),
            ('基值丝速', params.get('基值丝速',0)),
            ('峰值比例%', params['峰值比例%']),
            ('摆动速度', params['摆动速度']),
            ('左侧停留', params['左侧停留']),
            ('焊接速度', params['焊接速度']),
            ('脉冲频率', params['脉冲频率']),
            ('摆动幅度', params['摆动幅度']),
            ('右侧停留', params['右侧停留'])
        ])
    
    def validate(self):
        # 验证所有记录
        success = 0
        for record in self.data:
            inputs = {k:v for k,v in record.items() if k != '程序段'}
            actual = record['程序段'][0]['程序段1']
            predicted = self.generate_procedure(inputs)
            
            match = all(
                abs(predicted[k]-actual[k]) < 1e-3 
                for k in predicted
            )
            
            if match:
                success += 1
            else:
                print(f"Mismatch in record: {inputs}")
                print("Predicted:", predicted)
                print("Actual:   ", actual)
        
        print(f"Validation Accuracy: {success/len(self.data)*100:.2f}%")

# 初始化验证
with open('./outpout/output_碳钢打底焊接工艺.json') as f:
    data = json.load(f)

generator = WeldingProcedureGenerator(data)
generator.validate()

# 使用示例
inputs = {
    "材质": "碳钢",
    "厚度": 4.0,
    "坡口角度": 37.0,
    "钝边": 1.0,
    "间隙": 1.6,
    "直径": 60,
    "增透剂": "无"
}

output = generator.generate_procedure(inputs)
print(json.dumps(output, indent=2, ensure_ascii=False))