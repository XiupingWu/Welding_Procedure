# 焊接工艺参数预测系统

## 项目概述
此系统使用机器学习技术自动预测和生成焊接工艺参数，特别针对碳钢和不锈钢打底焊接工艺优化。系统通过分析历史焊接数据，构建精确的预测模型，为焊接工程师提供可靠的工艺参数建议。

## 主要功能
- **数据提取**：从Excel工艺文件中自动解析和标准化焊接工艺数据
- **参数预测**：基于材料特性（厚度、坡口角度、钝边等）预测最优焊接参数
- **精确建模**：为每个焊接参数单独训练专业模型，提高预测精度
- **特征工程**：自动创建派生特征以捕捉复杂的非线性关系
- **工艺验证**：提供模型性能验证和结果分析功能

## 系统架构

### 1. 数据处理模块 (ExcelDataExtractor.py)
- 从Excel文件中提取原始焊接工艺数据
- 处理数据缺失、异常和格式问题
- 标准化数据结构并转换为JSON格式
- 合并多程序段数据，计算参数平均值

### 2. 机器学习训练模块 (MLTrainer/WeldingProcedureMLTrainer.py)
- 数据预处理和特征工程
- 特征重要性评估和特征子集选择
- 多模型训练（决策树、随机森林、梯度提升树）
- 交叉验证和性能评估
- 模型导出和保存

### 3. 参数生成模块 (WeldingProcedureGenerator.py)
- 加载训练好的模型和特征映射
- 基于输入条件生成焊接参数预测
- 应用后处理规则优化预测结果
- 标准化输出格式

## 技术亮点
- **特征工程**：实现了多种高级特征转换（对数变换、二次项、交互特征等）
- **特征选择**：为每个焊接参数选择最佳特征子集，提高预测精度
- **多模型比较**：支持多种机器学习算法，可根据数据特性选择最优模型
- **参数优化**：应用领域知识实现预测结果的智能后处理调整

## 安装和使用

### 环境需求
- Python 3.6+
- 依赖库：numpy, pandas, scikit-learn, joblib

### 安装
```bash
# 克隆项目
git clone https://github.com/username/welding-procedure-training-model.git
cd welding-procedure-training-model

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install numpy pandas scikit-learn joblib
```

### 使用流程

1. **数据准备**：将焊接工艺Excel文件放入`excel/`目录

2. **数据提取**：
   ```bash
   python src/ExcelDataExtractor.py
   ```
   提取的数据将保存在`output/`目录

3. **模型训练**：
   ```bash
   python src/MLTrainer/WeldingProcedureMLTrainer.py
   ```
   训练后的模型将保存在`trained_models/`目录

4. **参数预测**：
   ```python
   from src.WeldingProcedureGenerator import WeldingProcedureGenerator
   
   # 初始化生成器
   generator = WeldingProcedureGenerator('./trained_models/model_package.pkl')
   
   # 定义输入参数
   input_params = {
       "材质": "碳钢",
       "厚度": 5.0,
       "坡口角度": 37.0,
       "钝边": 1.0,
       "间隙": 1.6,
       "直径": 60,
       "增透剂": "是"
   }
   
   # 生成焊接参数
   welding_params = generator.generate(input_params)
   print(welding_params)
   ```

## 最新更新
- 增强特征工程，添加交互特征和非线性变换
- 优化模型参数，提高预测准确度
- 改进后处理规则，更好地处理特殊工艺案例

## 维护与扩展
- 定期更新训练数据，保持模型准确性
- 添加新的材料和工艺类型支持
- 优化特征集和模型超参数
- 实现更多的后处理规则和模型融合策略

## 贡献
欢迎通过提交Issue或Pull Request为项目贡献代码和改进建议。

## 许可
[请补充适当的许可信息]