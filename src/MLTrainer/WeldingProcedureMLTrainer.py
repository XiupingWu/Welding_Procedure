import json
import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class WeldingModelTrainer:
    """
    焊接工艺参数机器学习模型训练器
    负责数据预处理、特征选择、模型训练和验证
    """
    def __init__(self, data_set):
        """
        初始化训练器实例
        
        参数:
        - data_set: 焊接工艺数据集列表
        """
        self.data = data_set;
        self.material_encoder = {} # Save the material list from data set (碳钢，不锈钢，etc),
        self.models = {}
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
        """
        将材质名称编码为数值表示
        将所有唯一的材质名称转换为整数索引
        """
        materials = list(set([d['材质'] for d in self.data]))
        self.material_encoder = {mat: idx for idx, mat in enumerate(materials)}

    def _build_feature_vector(self, record):
        """
        从单条焊接记录构建特征向量
        
        参数:
        - record: 单条焊接工艺记录
        
        返回:
        - 包含原始特征和派生特征的特征向量
        """
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

    def train_with_feature_selection(
            self, 
            save_path='./trained_models', 
            threshold=0., 
            do_validation=True, 
            model_type='dt', 
            n_jobs=-1
        ):
        """
        使用针对每个输出参数的最佳特征子集进行训练
        
        参数:
        - save_path: 模型保存路径
        - threshold: 特征重要性阈值
        - do_validation: 是否进行验证
        - model_type: 模型类型 ('dt'=决策树, 'rf'=随机森林, 'gbdt'=梯度提升树)
        - n_jobs: 并行作业数，-1表示使用所有CPU
        
        返回:
        - 为每个输出参数训练的专门模型字典
        """
        # 数据预处理
        self._encode_materials()
        
        # 准备所有特征的完整数据集
        full_X = []
        param_keys = [
                        '峰值电流','焊接速度','摆动速度',
                        '摆动幅度','左侧停留','右侧停留',
                        '脉冲频率','峰值丝速','峰值比例%'
                    ]

        self.y = {key: [] for key in param_keys}
        
        # 矢量化数据处理
        for record in self.data:
            full_X.append(self._build_feature_vector(record))
            segment = record['程序段'][0]
            for param in self.y:
                self.y[param].append(segment[param])
        
        full_X = np.array(full_X)
        for k in self.y:
            self.y[k] = np.array(self.y[k])
        
        # 暂时设置X以便分析特征重要性
        self.X = full_X
        
        # 创建模型工厂
        def create_model(model_type):
            """
            根据指定的模型类型创建回归模型实例
            
            参数:
            - model_type: 模型类型名称
            
            返回:
            - 配置好的回归模型实例
            """
            if model_type == 'dt':
                return DecisionTreeRegressor(min_samples_leaf=3)
            elif model_type == 'rf':
                return RandomForestRegressor(min_samples_leaf=3, n_estimators=100, n_jobs=1)
            elif model_type == 'gbdt':
                return GradientBoostingRegressor(min_samples_leaf=3, n_estimators=100)
            else:
                return DecisionTreeRegressor(min_samples_leaf=3)
        
        # 定义特征选择函数
        def select_features_for_param(param_name, y_values):
            """
            为特定输出参数选择最重要的特征子集
            
            参数:
            - param_name: 输出参数名称
            - y_values: 对应的输出参数值
            
            返回:
            - 参数名称、特征重要性列表、选中的特征索引、选中的特征名称
            """
            # 计算互信息
            mi_scores = mutual_info_regression(full_X, y_values)

            # 训练模型以获取特征重要性
            temp_model = create_model(model_type)
            temp_model.fit(full_X, y_values)
            tree_importance = temp_model.feature_importances_
            
            # 整合结果
            param_importance = []
            for i, feature_name in enumerate(self.feature_names):
                param_importance.append({
                    '特征名称': feature_name,
                    '互信息': mi_scores[i],
                    '模型重要性': tree_importance[i],
                    '索引': i
                })

            # 按互信息排序
            param_importance.sort(key=lambda x: x['互信息'], reverse=True)
            
            # 筛选出重要特征
            selected_indices = []
            selected_names = []
            
            for imp in param_importance:
                if imp['互信息'] > threshold or imp['模型重要性'] > threshold:
                    selected_indices.append(imp['索引'])
                    selected_names.append(imp['特征名称'])
            
            # 至少选择3个最重要的特征
            if len(selected_indices) < 3:
                selected_indices = [param_importance[i]['索引'] for i in range(min(3, len(param_importance)))]
                selected_names = [param_importance[i]['特征名称'] for i in range(min(3, len(param_importance)))]
            
            return param_name, param_importance, selected_indices, selected_names
        
        # 并行执行特征选择
        print("\n分析特征重要性...")
        selection_results = Parallel(n_jobs=n_jobs)(
            delayed(select_features_for_param)(param_name, y_values) 
            for param_name, y_values in self.y.items()
        )
        
        # 处理特征选择结果
        importance_data = {}
        feature_selections = {}
        
        for param_name, param_importance, selected_indices, selected_names in selection_results:
            importance_data[param_name] = param_importance
            feature_selections[param_name] = {
                'indices': selected_indices,
                'names': selected_names
            }
            
            # 打印特征重要性分析
            print(f"\n{param_name}的特征重要性分析:")
            print(f"{'特征名称':<15} {'互信息':<10} {'模型重要性':<10}")
            print("-" * 40)
            
            for imp in param_importance:
                print(f"{imp['特征名称']:<15} {imp['互信息']:.4f}    {imp['模型重要性']:.4f}")
            
            print(f"\n{param_name}选择的特征: {', '.join(selected_names)}")
        
        # 定义训练函数
        def train_model_for_param(param, selection):
            """
            使用选定的特征子集为特定参数训练模型
            
            参数:
            - param: 输出参数名称
            - selection: 包含特征选择信息的字典
            
            返回:
            - 参数名称和训练好的模型信息
            """
            indices = selection['indices']
            X_subset = full_X[:, indices]
            
            model = create_model(model_type)
            model.fit(X_subset, self.y[param])
            
            return param, {
                'model': model,
                'feature_indices': indices,
                'feature_names': selection['names']
            }
        
        # 并行训练模型
        print("\n训练专用模型...")
        model_results = Parallel(n_jobs=n_jobs)(
            delayed(train_model_for_param)(param, selection)
            for param, selection in feature_selections.items()
        )
        
        # 处理训练结果
        specialized_models = {param: model_info for param, model_info in model_results}
        
        # 保存模型
        os.makedirs(save_path, exist_ok=True)
        joblib.dump({
            'models': specialized_models,
            'material_encoder': self.material_encoder,
            'feature_names': self.feature_names,
            'model_type': model_type
        }, f'{save_path}/model_package.pkl')
        
        print(f"\n模型训练完成，已保存至 {save_path}")
        
        # 使用特征子集进行验证
        if do_validation:
            print("\n使用特征子集进行模型验证...")
            self.validate_with_feature_subsets(specialized_models, full_X)
        
        return specialized_models

    def validate_with_feature_subsets(self, specialized_models, full_X, n_splits=5):
        """
        使用特征子集执行K折交叉验证
        
        参数:
        - specialized_models: 为每个参数训练的专用模型字典
        - full_X: 完整特征矩阵
        - n_splits: K折交叉验证的折数
        
        返回:
        - 包含每个参数验证指标的字典
        """
        kf = KFold(n_splits=n_splits)
        
        metrics = {
            param: {'MAE': [], 'R2': []}
            for param in self.y
        }
        
        for train_idx, val_idx in kf.split(full_X):
            # 为每个参数训练和验证
            for param, model_info in specialized_models.items():
                # 提取该参数的特征子集
                indices = model_info['feature_indices']
                X_subset = full_X[:, indices]
                
                # 划分训练/验证集
                X_train = X_subset[train_idx]
                X_val = X_subset[val_idx]
                y_train = self.y[param][train_idx]
                y_val = self.y[param][val_idx]
                
                # 训练模型
                model = DecisionTreeRegressor(min_samples_leaf=3)
                model.fit(X_train, y_train)
                
                # 验证集预测
                y_pred = model.predict(X_val)
                
                # 计算指标
                metrics[param]['MAE'].append(
                    mean_absolute_error(y_val, y_pred)
                )
                metrics[param]['R2'].append(
                    r2_score(y_val, y_pred)
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
            
            print(f"\n{param}验证结果 (使用特征子集):")
            print(f"特征子集: {', '.join(specialized_models[param]['feature_names'])}")
            print(f"MAE: {final_metrics[param]['MAE_mean']:.2f} ± {final_metrics[param]['MAE_std']:.2f}")
            print(f"R²: {final_metrics[param]['R2_mean']:.2f} ± {final_metrics[param]['R2_std']:.2f}")
        
        return final_metrics

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
    trainer.train_with_feature_selection();

if __name__ == '__main__':
    main();
