import json
import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

class WeldingModelTrainer:
    def __init__(self, data_set):
        self.data = data_set;
        self.material_encoder = {} # Save the material list from data set (碳钢，不锈钢，etc)
        self.feature_names = [
            '厚度', 
            '坡口角度', 
            '钝边', 
            '间隙', 
            '直径', 
            '增透剂', 
            '材质',
            '厚度×坡口角度',  # 新增派生特征名称
            '归一化直径', 
            '直径对数', 
            '厚度平方', 
            '钝边比例'
        ]
        self.X = None;
        self.y = None;
    
    
    def _encode_materials(self):
        materials = list(set([d['材质'] for d in self.data]))
        self.material_encoder = {mat: idx for idx, mat in enumerate(materials)}
    
    def _build_feature_vector(self, record):
        features =  [
            record['厚度'],
            record['坡口角度'],
            record['钝边'],
            record['间隙'],
            record['直径'],
            1 if record['增透剂'] == '是' else 0,
            self.material_encoder[record['材质']]
        ]

        # 添加派生特征
        features.extend([
            record['厚度'] * record['坡口角度'],  # 相互作用特征
            record['直径'] / 100.0,  # 归一化直径
            np.log1p(record['直径']),  # 对数变换
            record['厚度'] ** 2,  # 二次项
            record['钝边'] / (record['厚度'] + 0.001)  # 钝边比例
        ]);
        
        return features;
    
    def validate(self, n_splits=5, verbose=True):
        """执行K折交叉验证并返回指标"""
        kf = KFold(n_splits=n_splits)
        metrics = {
            param: {'MAE': [], 'R2': []}
            for param in self.y
        }
        
        for train_idx, val_idx in kf.split(self.X):
            # 划分训练/验证集
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            
            fold_models = {}
            # 训练临时模型
            for param in self.y:
                model = DecisionTreeRegressor(min_samples_leaf=3)
                model.fit(X_train, self.y[param][train_idx])
                fold_models[param] = model
            
            # 验证集预测
            for param, model in fold_models.items():
                y_pred = model.predict(X_val)
                y_true = self.y[param][val_idx]
                
                # 计算指标
                metrics[param]['MAE'].append(
                    mean_absolute_error(y_true, y_pred)
                )
                metrics[param]['R2'].append(
                    r2_score(y_true, y_pred)
                )
        
        # 汇总结果
        final_metrics = {}
        for param in metrics:
            final_metrics[param] = {
                'MAE_mean': np.mean(metrics[param]['MAE']),
                'MAE_std': np.std(metrics[param]['MAE']),
                'R2_mean': np.mean(metrics[param]['R2']),
                'R2_std': np.std(metrics[param]['R2'])
            }
            
            if verbose:
                print(f"\n{param}验证结果:")
                print(f"MAE: {final_metrics[param]['MAE_mean']:.2f} ± {final_metrics[param]['MAE_std']:.2f}")
                print(f"R²: {final_metrics[param]['R2_mean']:.2f} ± {final_metrics[param]['R2_std']:.2f}")
        
        return final_metrics
    
    def train(self, save_path='./trained_models', do_validation=True):
        # 数据预处理
        self._encode_materials();

        self.X = []
        self.y = {
            key: [] for key in [
                '峰值电流','焊接速度','摆动速度',
                '摆动幅度','左侧停留','右侧停留',
                '脉冲频率','峰值丝速','峰值比例%'
            ]
        }

        for record in self.data:
            self.X.append(self._build_feature_vector(record))
            segment = record['程序段'][0]
            for param in self.y:
                self.y[param].append(segment[param])
        
        self.X = np.array(self.X)
        
        for k in self.y:
            self.y[k] = np.array(self.y[k])
        
        # 训练最终模型
        models = {}
        for param in self.y:
            model = DecisionTreeRegressor(min_samples_leaf=3)
            model.fit(self.X, self.y[param]);
            '''对程序段内各个参数作为y，输入参数作为X进行回归预测'''
            models[param] = model
        
        # 执行验证
        if do_validation:
            print("\n开始模型验证...")
            print(f'包含材质： {self.material_encoder}')
            val_results = self.validate(verbose=True)
            
            # 关键参数阈值检查
            critical_params = {
                '峰值电流': {'MAE_threshold': 5.0, 'R2_threshold': 0.85},
                '焊接速度': {'MAE_threshold': 3.0, 'R2_threshold': 0.8}
            }
            
            for param, thresholds in critical_params.items():
                mae = val_results[param]['MAE_mean']
                r2 = val_results[param]['R2_mean']
                
                if mae > thresholds['MAE_threshold']:
                    print(f"警告: {param}的MAE {mae:.2f}超过阈值 {thresholds['MAE_threshold']}")
                if r2 < thresholds['R2_threshold']:
                    print(f"警告: {param}的R² {r2:.2f}低于阈值 {thresholds['R2_threshold']}")
        
        # 保存模型
        joblib.dump({
            'models': models,
            'material_encoder': self.material_encoder,
            'feature_names': self.feature_names,
            'validation_results': val_results if do_validation else None
        }, f'{save_path}/model_package.pkl')
        
        print(f"\n模型训练完成，已保存至 {save_path}")

def main():
    # 初始化验证
    json_dir = './output';

    # 引入全部json file
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    data_set = [];
    for file in json_files:
        with open(os.path.join(json_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_set.extend(data)
    
    # 训练阶段（定期执行）
    trainer = WeldingModelTrainer(data_set);
    trainer.train();

if __name__ == '__main__':
    main();
