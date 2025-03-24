import math
from typing import Dict, List, Union

class CarbonSteelWeldProcedure:
    """
    碳钢焊接工艺生成器
    :param weld_attrs: 焊件属性字典
        - thickness: 厚度(mm)
        - groove_angle: 坡口角度(度) 
        - root_face: 钝边(mm)
        - gap: 间隙(mm)
        - diameter: 直径(mm)
    :param machine_params: 焊机设定参数
        - ignition_current: 起弧电流(A)
        - preheat_time: 预热时间(s)
        - arc_length: 拖弧长度(mm)
        - torch_height: 提枪高度(mm)
        - wire_pulling: 是否抽丝(bool)
    :param penetration_aid: 是否使用增透剂(bool)
    """
    
    def __init__(self, 
                weld_attrs: Dict[str, float], 
                machine_params: Dict[str, Union[float, bool]], 
                penetration_aid: bool):
        
        # 参数校验
        self._validate_input(weld_attrs, machine_params)
        
        # 绑定参数
        self.weld_attrs = weld_attrs
        self.machine_params = machine_params
        self.penetration_aid = penetration_aid
        
        # 计算核心参数
        self._procedure = self._calculate_procedure()

    def get_procedure(self) -> Dict:
        return self._procedure;

    def _validate_input(self, attrs: dict, params: dict):
        """输入参数校验"""
        required_attrs = ['thickness', 'groove_angle', 'root_face', 'gap', 'diameter']
        required_params = ['ignition_current', 'preheat_time', 'arc_length', 'torch_height', 'wire_pulling']
        
        for key in required_attrs:
            if key not in attrs:
                raise ValueError(f"缺少焊件属性: {key}")
                
        for key in required_params:
            if key not in params:
                raise ValueError(f"缺少焊机参数: {key}")

    def _calculate_core_params(self) -> Dict[str, float]:
        """计算核心焊接参数"""
        t = self.weld_attrs['thickness']
        D = self.weld_attrs['diameter']
        theta = self.weld_attrs['groove_angle']
        g = self.weld_attrs['gap']
        r = self.weld_attrs['root_face']
        
        # 计算模式选择
        if self.penetration_aid:
            # TAW模式计算公式
            Ip = (18 * math.pow(t, 1.1) + 7 * math.pow(D, 0.6))
            Ip *= math.pow(0.85, g + 0.2*r)  # 钝边补偿
            v_wire = 110 * math.pow(Ip, 0.7) * (1 + 0.015*theta)
            seg_count = math.ceil(0.03*D + 0.2*t)
        else:
            # CTAW模式计算公式
            Ip = (22 * math.pow(t, 0.9) + 10 * math.sqrt(D))
            Ip *= math.pow(0.7, g + 0.3*r)  # 钝边补偿
            exp_factor = 0.008 if theta > 35 else 0.01
            v_wire = 0.9 * (90 * math.pow(Ip, 0.8) * math.exp(exp_factor*theta))
            seg_count = math.ceil((0.04*D + 0.25*t + 0.1*theta) * (1 - D/1000))
        
        # 约束处理
        seg_count = max(1, min(5, seg_count))
        v_wire = round(v_wire / 10) * 10  # 取整到10的倍数
        
        return {
            'peak_current': round(Ip, 1),
            'wire_speed': v_wire,
            'segment_count': seg_count,
            'travel_speed': self._calculate_travel_speed(Ip, D, t)
        }

    def _calculate_travel_speed(self, Ip: float, D: float, t: float) -> float:
        """计算焊接速度"""
        if self.penetration_aid:
            # TAW模式速度公式
            return round((1.2 * Ip) / t * 1000 / 60, 1)  # 单位转换：m/h → mm/min
        else:
            # CTAW模式速度公式
            return round((0.8 * Ip * D) / 1000 * 1000 / 60, 1)

    def _build_segments(self, core_params: dict) -> List[Dict]:
        """构建程序段数据结构"""
        segments = []
        base_current = round(core_params['peak_current'] * 0.6, 1)
        
        for seg_num in range(1, core_params['segment_count'] + 1):
            # 生成摆动参数
            oscillation = {
                'amplitude': round(1.2*self.weld_attrs['gap'] + 0.5*self.weld_attrs['thickness'], 1),
                'frequency': round(1.5 - 0.05*self.weld_attrs['thickness'], 1),
                'dwell_time': round(0.08*self.weld_attrs['thickness'], 1),
                'left_dwell': None,  # 预留字段
                'right_dwell': None
            }
            
            # 程序段参数
            segment = {
                'segment_number': seg_num,
                'peak_current': core_params['peak_current'],
                'base_current': base_current,
                'wire_speed': core_params['wire_speed'],
                'travel_speed': core_params['travel_speed'],
                'oscillation': oscillation,
                'special_params': None  # 预留扩展字段
            }
            segments.append(segment)
        
        return segments

    def _calculate_procedure(self) -> Dict:
        """生成完整工艺规程"""
        core_params = self._calculate_core_params()
        
        return {
            'material': '碳钢',
            'weld_attributes': self.weld_attrs,
            'machine_settings': self.machine_params,
            'penetration_aid': self.penetration_aid,
            'segments': self._build_segments(core_params),
            'calculated_params': core_params
        }

    def validate(self, expected_data: Dict) -> Dict:
        """工艺验证
        :param expected_data: 包含预期结果的字典
        :return: 验证报告
        """
        report = {
            'segment_count_match': len(self.procedure['segments']) == expected_data['segment_count'],
            'param_errors': {}
        }
        
        # 参数误差计算
        for param in ['peak_current', 'wire_speed', 'travel_speed']:
            gen_val = self.procedure['calculated_params'][param]
            exp_val = expected_data[param]
            error = abs(gen_val - exp_val) / exp_val * 100
            report['param_errors'][param] = f"{error:.2f}%"
        
        return report

# 测试示例
if __name__ == "__main__":
    # 测试案例配置
    test_case = {
        'input': {
            'weld_attrs': {
                'thickness': 10.0,
                'groove_angle': 37.5,
                'root_face': 0,
                'gap': 2.5,
                'diameter': 108
            },
            'machine_params': {
                'ignition_current': 70,
                'preheat_time': 8,
                'arc_length': 15,
                'torch_height': 10,
                'wire_pulling': True
            },
            'penetration_aid': False
        }
    }

    # 执行测试
    carbonSteel_weldingProcedure = CarbonSteelWeldProcedure(**test_case['input']);
    output = carbonSteel_weldingProcedure.get_procedure();
    print(output);
    # validation_report = generator.validate(test_case['expected'])
    
    # # 输出结果
    # print("生成的焊接工艺：")
    # print(json.dumps(generator.procedure, indent=2, ensure_ascii=False))
    
    # print("\n验证报告：")
    # print(json.dumps(validation_report, indent=2, ensure_ascii=False))