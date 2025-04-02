import math
import pandas as pd
import os
import json

def safe_convert(value, target_type):
    if not pd.isna(value):
        try:
            return target_type(value);
        except (ValueError, TypeError):
            return 0;
    else:
        return 0;

def parse_welding_data(file_path, sheetNum = 0):
    # 读取Excel文件，跳过首行标题并禁用自动列名
    raw_df = pd.read_excel(file_path, header=None, sheet_name=sheetNum);
    
    # 存储所有工艺组的结果
    results = []
    # 当前正在处理的工艺组
    current_group = None
    
    for row_index, row in raw_df.iterrows():
        # 获取下一行数据，用于解析工艺参数
        next_row = raw_df.iloc[row_index + 1] if row_index + 1 < len(raw_df) else None

        # 检测新工艺组起始标记
        if row[0] == '材质':
            # 初始化新的工艺组数据结构
            current_group = {
                '材质': next_row[0],  # 材料类型
                '厚度': safe_convert(next_row[1], float),  # 材料厚度(mm)
                '坡口角度': safe_convert(next_row[2], float),  # 坡口角度(°)
                '钝边': safe_convert(next_row[3], float),  # 钝边尺寸(mm)
                '间隙': safe_convert(next_row[4], float),  # 焊缝间隙(mm)
                '直径': None,  # 焊丝直径(mm)
                '增透剂': next_row[5] if not pd.isna(next_row[5]) else None,  # 是否使用增透剂
                '程序段': []  # 焊接程序段列表
            }
            results.append(current_group)
        
        # 焊丝配置解析
        elif row[1] == '直径':
            current_group['直径'] = safe_convert(next_row[1], int)

        # 程序段解析
        elif isinstance(row[0], str) and '程序段' in row[0]:
            # 初始化程序段数据结构
            segment_data = {
                '焊接角度': 0,  # 焊接角度(°)
                '峰值电流': 0,  # 峰值电流(A)
                '峰值比例%': 0,  # 峰值（电流，丝速）占比(%)， 100%时基值为空，0%时峰值为空
                '峰值丝速': 0,  # 峰值送丝速度(mm/s)
                '摆动速度': 0,  # 焊枪摆动速度(mm/s)
                '左侧停留': 0,  # 左侧停留时间(s)
                '焊接速度': 0,  # 焊接速度(mm/min)
                '基值电流': 0,  # 基值电流(A)
                '脉冲频率': 0,  # 脉冲频率(Hz)
                '基值丝速': 0,  # 基值送丝速度(mm/s)
                '摆动幅度': 0,  # 焊枪摆动幅度(mm)
                '右侧停留': 0  # 右侧停留时间(s)
            }

            for i in range(row_index, row_index + 4):
                if i >= len(raw_df):
                    break;
                
                segment_row = raw_df.iloc[i];
                segment_row_next_row = raw_df.iloc[i + 1] if i + 1 < len(raw_df) else None;

                if segment_row_next_row is None:
                    break;
                
                if segment_row[1] == '焊接角度':
                    if pd.isna(segment_row_next_row[1]):
                        break;
                    else:
                        segment_data['焊接角度'] = safe_convert(segment_row_next_row[1], int)
                        segment_data['峰值电流'] = safe_convert(segment_row_next_row[2], int);
                        segment_data['峰值比例%'] = safe_convert(segment_row_next_row[3], int);
                        segment_data['峰值丝速'] = safe_convert(segment_row_next_row[4], int);
                        segment_data['摆动速度'] = safe_convert(segment_row_next_row[5], float);
                        segment_data['左侧停留'] = safe_convert(segment_row_next_row[6], float);

                elif segment_row[1] == '焊接速度':
                    segment_data['焊接速度'] = safe_convert(segment_row_next_row[1], int);
                    segment_data['基值电流'] = safe_convert(segment_row_next_row[2], int);
                    segment_data['脉冲频率'] = safe_convert(segment_row_next_row[3], int);
                    segment_data['基值丝速'] = safe_convert(segment_row_next_row[4], int);
                    segment_data['摆动幅度'] = safe_convert(segment_row_next_row[5], int);
                    segment_data['右侧停留'] = safe_convert(segment_row_next_row[6], float);
            
            if not pd.isna(segment_data['焊接角度']):
                current_group['程序段'].append(segment_data)  # 直接添加segment_data
            row_index+=3;

            # 计算所有程序段的焊接角度之和
            total_angle = sum(x['焊接角度'] for x in current_group['程序段'])
            
            # 计算各参数的平均值，忽略0值，并向上取整
            def avg_ignore_zero(values):
                non_zero_values = [v for v in values if v != 0]
                return math.ceil(sum(non_zero_values) / len(non_zero_values)) if non_zero_values else 0
            
            avg_peak_current = avg_ignore_zero(x['峰值电流'] for x in current_group['程序段'])
            avg_base_current = avg_ignore_zero(x['基值电流'] for x in current_group['程序段'])
            avg_peak_wire_speed = avg_ignore_zero(x['峰值丝速'] for x in current_group['程序段'])
            avg_base_wire_speed = avg_ignore_zero(x['基值丝速'] for x in current_group['程序段'])
            
            # 创建新的综合程序段
            combined_segment = {
                '焊接角度': total_angle,
                '峰值电流': avg_peak_current,
                '基值电流': avg_base_current,
                '峰值丝速': avg_peak_wire_speed,
                '基值丝速': avg_base_wire_speed,
                # 其他参数保持不变，取第一个程序段的值
                '峰值比例%': current_group['程序段'][0]['峰值比例%'],
                '摆动速度': current_group['程序段'][0]['摆动速度'],
                '左侧停留': current_group['程序段'][0]['左侧停留'],
                '焊接速度': current_group['程序段'][0]['焊接速度'],
                '脉冲频率': current_group['程序段'][0]['脉冲频率'],
                '摆动幅度': current_group['程序段'][0]['摆动幅度'],
                '右侧停留': current_group['程序段'][0]['右侧停留']
            }
            
            current_group['程序段'] = [combined_segment]
    
    # 移除全部丝速为0的数据
    indices_to_remove = []

    for index, el in enumerate(results):
        if el['程序段'][0]['峰值丝速'] == 0 and el['程序段'][0]['基值丝速']==0:
            indices_to_remove.append(index)

    # 从后往前删除
    for index in sorted(indices_to_remove, reverse=True):
        del results[index]

    return results;

# 使用main函数封装主逻辑
def main():

    main_output_dir = './output';
    os.makedirs(main_output_dir, exist_ok=True);
    # 读取Excel文件
    excel_file = './excel/碳钢打底焊接工艺.xlsx'
    
    # 获取所有工作表名称
    sheets = pd.ExcelFile(excel_file).sheet_names
    sheetName = os.path.splitext(os.path.basename(excel_file))[0]
    outputSheets = [];
    # 遍历每个工作表
    for index in range(len(sheets)):
        # 解析当前工作表数据
        sheet_data = parse_welding_data(excel_file, index);
        outputSheets+=sheet_data;
    print(len(outputSheets))
    outputSheets_json = json.dumps(outputSheets, indent=2, ensure_ascii=False)
    # 生成以工作表名称命名的JSON文件
    output_file = f'{main_output_dir}/output_{sheetName}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(outputSheets_json)

if __name__ == '__main__':
    main()