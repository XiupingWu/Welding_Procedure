import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

class WeldingProcedureTrainer:
    def __init__(self, data, model_path='./TrainedModels/welding_procedure_models.joblib'):
        self.data = data
        self.material_map = self._create_material_map();
        self.model_path = model_path    
        # 训练模型
        self.models = self._train_models()
    
    def validate(self):
        # 验证所有记录
        success = 0
        for record in self.data:
            inputs = {k:v for k,v in record.items() if k != '程序段'}
            actual = record['程序段'][0]
            predicted = self.generate_procedure(inputs)
            
            match = all(
                abs(predicted[k]-actual[k]) < 1e-3 
                for k in predicted
            )
            
            if match:
                success += 1
            else:
                print(f"Mismatch in record: {inputs} \n")
                print(f"Predicted: {predicted} \n")
                print(f"Actual: {actual} \n")
        
        print(f"Validation Accuracy: {success/len(self.data)*100:.2f}%");
    
    def _train_models(self):
        # 准备训练数据
        X, y = self._prepare_training_data()
        
        # 训练模型
        params = [
            '峰值电流', '焊接速度', '摆动速度', '摆动幅度', 
            '左侧停留', '右侧停留', '脉冲频率', '峰值丝速'
        ]
        
        models = {}
        for param in params:
            model = DecisionTreeRegressor()
            model.fit(X, y[param])
            models[param] = model
        
        # 保存模型
        dump(models, self.model_path)
        print(f"Models saved to {self.model_path}")
        return models
    
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
            
            segment = record['程序段'][0]
                
            for param in y:
                y[param].append(segment[param])
                
            X.append(features)
            
        return np.array(X), y

def main():
    # 初始化验证
    with open('./outpout/output_碳钢打底焊接工艺.json') as f:
        data = json.load(f)

    generator = WeldingProcedureTrainer(data);
    generator.validate();

if __name__ == '__main__':
    main();
