import pandas as pd
import json

def safe_convert(value, target_type):
    if not pd.isna(value):
        try:
            return target_type(value);
        except (ValueError, TypeError):
            return None;
    else:
        return;

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
                '厚度': safe_convert(next_row[1], int),  # 材料厚度(mm)
                '坡口角度': safe_convert(next_row[2], float),  # 坡口角度(°)
                '钝边': safe_convert(next_row[3], float),  # 钝边尺寸(mm)
                '间隙': safe_convert(next_row[4], float),  # 焊缝间隙(mm)
                '增透剂': next_row[5] if not pd.isna(next_row[5]) else None,  # 是否使用增透剂
                '直径': None,  # 焊丝直径(mm)
                '起弧电流': None,  # 起弧电流(A)
                '预热时间': None,  # 预热时间(s)
                '拖弧长度': None,  # 拖弧长度(mm)
                '提枪高度': None,  # 提枪高度(mm)
                '是否抽丝': None,  # 是否抽丝
                '程序段': []  # 焊接程序段列表
            }
            results.append(current_group)
        
        # 焊丝配置解析
        elif row[1] == '直径':
            current_group['直径'] = safe_convert(next_row[1], int)
            current_group['起弧电流'] = safe_convert(next_row[2], int)
            current_group['预热时间'] = safe_convert(next_row[3], int)
            current_group['拖弧长度'] = safe_convert(next_row[4], int)
            current_group['提枪高度'] = safe_convert(next_row[5], int)
            current_group['是否抽丝'] = bool(safe_convert(row[6], int))

        # 程序段解析
        elif isinstance(row[0], str) and '程序段' in row[0]:
            # 初始化程序段数据结构
            segment_data = {
                '焊接角度': None,  # 焊接角度(°)
                '峰值电流': None,  # 峰值电流(A)
                '峰值比例': None,  # 峰值电流比例(%)
                '峰值丝速': None,  # 峰值送丝速度(mm/s)
                '摆动速度': None,  # 焊枪摆动速度(mm/s)
                '左侧停留': None,  # 左侧停留时间(s)
                '焊接速度': None,  # 焊接速度(mm/min)
                '基值电流': None,  # 基值电流(A)
                '脉冲频率': None,  # 脉冲频率(Hz)
                '基值丝速': None,  # 基值送丝速度(mm/s)
                '摆动幅度': None,  # 焊枪摆动幅度(mm)
                '右侧停留': None  # 右侧停留时间(s)
            }

            for i in range(row_index, row_index + 4):
                if i >= len(raw_df):
                    break;
                
                segment_row = raw_df.iloc[i];
                segment_row_next_row = raw_df.iloc[i + 1] if i + 1 < len(raw_df) else None;

                if segment_row_next_row is None:
                    break;
                
                if segment_row[1] == '焊接角度':
                    segment_data['焊接角度'] = safe_convert(segment_row_next_row[1], int);
                    segment_data['峰值电流'] = safe_convert(segment_row_next_row[2], int);
                    segment_data['峰值比例'] = safe_convert(segment_row_next_row[3], int);
                    segment_data['峰值丝速'] = safe_convert(segment_row_next_row[4], int);
                    segment_data['摆动速度'] = safe_convert(segment_row_next_row[5], int);
                    segment_data['左侧停留'] = safe_convert(segment_row_next_row[6], int);

                elif segment_row[1] == '焊接速度':
                    segment_data['焊接速度'] = safe_convert(segment_row_next_row[1], int);
                    segment_data['基值电流'] = safe_convert(segment_row_next_row[2], int);
                    segment_data['脉冲频率'] = safe_convert(segment_row_next_row[3], int);
                    segment_data['基值丝速'] = safe_convert(segment_row_next_row[4], int);
                    segment_data['摆动幅度'] = safe_convert(segment_row_next_row[5], int);
                    segment_data['右侧停留'] = safe_convert(segment_row_next_row[6], int);

            current_group['程序段'].append({row[0] : segment_data});
            row_index+=3;

    return json.dumps(results, indent=2, ensure_ascii=False)

# 使用main函数封装主逻辑
def main():
    carbonSteel_pentration_aid = parse_welding_data('./excel/碳钢打底焊接工艺.xlsx', 0)
    with open('output.json', 'w', encoding='utf-8') as f:
        f.write(carbonSteel_pentration_aid)

if __name__ == '__main__':
    main()