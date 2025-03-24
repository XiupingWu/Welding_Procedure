import pandas as pd
import json
from collections import defaultdict

def safe_convert(value, target_type):
    if not pd.isna(value):
        try:
            return target_type(value);
        except (ValueError, TypeError):
            return None;
    else:
        return;

def parse_welding_data(file_path):
    # 跳过首行标题并禁用自动列名
    raw_df = pd.read_excel(file_path, header=None)
    
    results = []
    current_group = None
    
    for _, row in raw_df.iterrows():
        
        next_row = raw_df.iloc[_ + 1] if _ + 1 < len(raw_df) else None;

        # 检测新工艺组起始标记
        if row[0] == '材质':
            current_group = {
                '材质': next_row[0],
                '厚度': safe_convert(next_row[1], int),
                '坡口角度': safe_convert(next_row[2], float),
                '钝边': safe_convert(next_row[3], float),
                '间隙': safe_convert(next_row[4], float),
                '增透剂': next_row[5] if not pd.isna(next_row[5]) else None,
                '直径': None,
                '起弧电流': None,
                '预热时间': None,
                '拖弧长度': None,
                '提枪高度': None,
                '是否抽丝': None,
                '程序段': []
            }
            results.append(current_group)
        
        # 焊丝配置解析（列索引需根据实际表格调整）
        elif row[1] == '直径':
            current_group['直径'] = safe_convert(next_row[1], int);
            current_group['起弧电流'] = safe_convert(next_row[2], int);
            current_group['预热时间'] = safe_convert(next_row[3], int);
            current_group['拖弧长度'] = safe_convert(next_row[4], int);
            current_group['提枪高度'] = safe_convert(next_row[5], int);
            current_group['是否抽丝'] = bool(safe_convert(row[6], int));
        # 程序段解析逻辑...

        elif isinstance(row[0], str) and '程序段' in row[0]:

            segment_data = {
                '焊接角度': None,
                '峰值电流': None,
                '峰值比例': None,
                '峰值丝速': None,
                '摆动速度': None,
                '左侧停留': None,
                '焊接速度': None,
                '基值电流': None,
                '脉冲频率': None,
                '基值丝速': None,
                '摆动幅度': None,
                '右侧停留': None
            }

            for i in range(_, _ + 4):
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
            _+=3;

    return json.dumps(results, indent=2, ensure_ascii=False)
# 使用示例
json_output = parse_welding_data('./碳钢打底焊接工艺.xlsx')
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_output)