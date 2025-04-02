import json
from pathlib import Path
from typing import OrderedDict;
from joblib import dump, load
import numpy as np

class WeldingProceduregenerator():
  def __init__(self, model_path='./TrainedModels/welding_procedure_models.joblib'):
    self.model_path = model_path;
    self.model = self._load_trained_model();
    # self._special_cases = self._identify_special_cases();

    # 特殊参数规则
    self.thickness_bins = np.array([3,5,8,12,25]);
    self.diameter_bins = np.array([50,100,200,400,1000]);

  def _load_trained_model(self):
    if Path(self.model_path).exists():
      print("Loading pre-trained models...")
      return load(self.model_path);
    else:
      raise FileNotFoundError("未找到预训练模型，请确保模型路径正确");

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
    ]);

  def _calculate_peak_ratio(self, inputs):
        # 峰值比例决策规则
        if inputs['间隙'] > 2 or inputs['坡口角度'] > 35:
            return 100 if inputs['增透剂'] == '是' else 80
        return 100 if inputs['厚度'] < 8 else 50
  
  def _calculate_welding_angle(self, inputs):
        # 焊接角度计算规则
        base_angle = 360 + (inputs['直径'] / 50)
        adjustment = inputs['厚度'] * 0.5 - inputs['坡口角度'] * 0.2
        return round(base_angle + adjustment, 1)
    
  def _identify_special_cases(self):
      # 识别特殊工艺参数组合
      specials = {}
      for record in self.data:
          key = (record['厚度'], record['坡口角度'], 
                  record['钝边'], record['间隙'], 
                  record['直径'], record['增透剂'])
          specials[key] = record['程序段'][0]
      return specials;

  def generate_procedure(self, inputs):
        # 检查特殊案例
        key = (inputs['厚度'], inputs['坡口角度'], inputs['钝边'],
                inputs['间隙'], inputs['直径'], inputs['增透剂']);
        
        # ToDo: Apply special cases checker
        # if key in self._special_cases:
        #     return self._format_output(self._special_cases[key])
        
        # 特征向量构建
        X = np.array([
                    [inputs['厚度'], inputs['坡口角度'], inputs['钝边'],
                    inputs['间隙'], inputs['直径'], 
                    1 if inputs['增透剂'] == '是' else 0]])
        
        # 参数预测
        params = {}
        for name, model in self.model.items():
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

def main():
    generator = WeldingProceduregenerator();

    inputs = {
        "材质": "碳钢",
        "厚度": 6.0,
        "坡口角度": 30.0,
        "钝边": 1.0,
        "间隙": 1.0,
        "直径": 60,
        "增透剂": "无",
    }

    output = generator.generate_procedure(inputs)
    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main();