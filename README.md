# 焊接工艺参数生成系统

## 项目概述
本系统用于自动生成碳钢打底焊接工艺参数，包含以下主要功能：
1. 从Excel文件中提取焊接工艺数据
2. 使用机器学习模型预测焊接参数
3. 生成标准化的焊接工艺程序
4. 验证模型预测准确性

## 系统架构
系统由两个主要模块组成：

### 1. 数据提取模块 (ExcelDataExtractor.py)
- 从Excel文件中读取焊接工艺数据
- 解析并转换数据格式
- 输出标准化的JSON格式数据

### 2. 工艺生成模块 (LLMCarbonSteelProcedure.py)
- 使用决策树回归模型预测焊接参数
- 处理特殊工艺案例
- 生成完整的焊接工艺程序
- 提供模型验证功能

## 主要功能
- **数据解析**：支持从Excel文件中提取焊接工艺参数
- **参数预测**：基于材料厚度、坡口角度等特征预测焊接参数
- **特殊案例处理**：识别并处理特殊工艺组合
- **模型验证**：验证预测结果与实际数据的匹配度

## 使用说明
1. 准备Excel数据文件，格式参考`./excel/碳钢打底焊接工艺.xlsx`
2. 运行数据提取模块：
   ```bash
   python excelDataExtractor.py
   ```
3. 使用工艺生成模块：
   ```python
   generator = WeldingProcedureGenerator(data)
   output = generator.generate_procedure(inputs)
   ```
4. 验证模型准确性：
   ```python
   generator.validate()
   ```

## 输入输出
- 输入：Excel文件，包含材料参数和焊接工艺数据
- 输出：JSON格式的标准化焊接工艺参数

## 依赖库
- numpy
- pandas
- scikit-learn

## 注意事项
1. 确保Excel文件格式符合要求
2. 模型训练数据需要定期更新
3. 特殊工艺案例需要手动维护

## 示例
```python
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
```

## 维护
- 定期更新训练数据
- 优化模型参数
- 添加新的特殊工艺案例


# Welding Procedure Generation System

## Project Overview
This system is designed to automatically generate welding parameters for carbon steel root welding, including the following main features:
1. Extract welding process data from Excel files
2. Predict welding parameters using machine learning models
3. Generate standardized welding procedures
4. Validate model prediction accuracy

## System Architecture
The system consists of two main modules:

### 1. Data Extraction Module (excelDataExtractor.py)
- Read welding process data from Excel files
- Parse and transform data format
- Output standardized JSON format data

### 2. Procedure Generation Module (LLMCarbonSteelProcedure.py)
- Predict welding parameters using Decision Tree Regression models
- Handle special process cases
- Generate complete welding procedures
- Provide model validation functionality

## Main Features
- **Data Parsing**: Extract welding parameters from Excel files
- **Parameter Prediction**: Predict welding parameters based on material thickness, groove angle, etc.
- **Special Case Handling**: Identify and process special process combinations
- **Model Validation**: Validate the accuracy of prediction results against actual data

## Usage Instructions
1. Prepare Excel data file, refer to `./excel/碳钢打底焊接工艺.xlsx` for format
2. Run data extraction module:
   ```bash
   python excelDataExtractor.py
   ```
3. Use procedure generation module:
   ```python
   generator = WeldingProcedureGenerator(data)
   output = generator.generate_procedure(inputs)
   ```
4. Validate model accuracy:
   ```python
   generator.validate()
   ```

## Input/Output
- Input: Excel file containing material parameters and welding process data
- Output: Standardized welding parameters in JSON format

## Dependencies
- numpy
- pandas
- scikit-learn

## Notes
1. Ensure Excel file format meets requirements
2. Model training data needs regular updates
3. Special process cases require manual maintenance

## Example
```python
inputs = {
    "Material": "Carbon Steel",
    "Thickness": 4.0,
    "Groove Angle": 37.0,
    "Blunt Edge": 1.0,
    "Gap": 1.6,
    "Diameter": 60,
    "Penetration Aid": "None"
}
output = generator.generate_procedure(inputs)
```

## Maintenance
- Regularly update training data
- Optimize model parameters
- Add new special process cases 