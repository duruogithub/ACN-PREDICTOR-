from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 定义阈值和模型路径
THRESHOLD = float(os.getenv("THRESHOLD", 0.136868298))
# 获取当前文件夹路径
CURRENT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURRENT_DIR, "rf_model.pkl")

# 加载模型
try:
    logger.info(f"正在加载模型文件：{MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, 'predict'):
        raise ValueError("加载的模型无效，请确保模型文件正确！")
except FileNotFoundError:
    raise FileNotFoundError(f"模型文件未找到，请确认路径：{MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"加载模型时发生未知错误: {e}")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()  # 获取完整错误堆栈
        logger.error(f"模板渲染错误：{error_details}")
        return f"模板渲染错误：{e}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """处理前端请求并返回预测结果"""
    try:
        # 提取表单数据并验证
        gender = int(request.form.get('gender', 0))
        age = int(request.form.get('age', 0))
        bmi = float(request.form.get('bmi', 0.0))
        residence = int(request.form.get('residence', 0))
        fx = int(request.form.get('fx', 0))
        bm = int(request.form.get('bm', 0))
        lwy = int(request.form.get('lwy', 0))
        smoke = int(request.form.get('smoke', 0))
        drink = int(request.form.get('drink', 0))
        fit = int(request.form.get('fit', 0))

        input_data = np.array([[gender, age, bmi, residence, fx, bm, lwy, smoke, drink, fit]])

        # 模型预测
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[0, 1]
            risk = probability * 100
            level, recommendation = ("高风险", "风险非常高！建议立即检查。") if probability > 0.9 else \
                                    ("中等风险", "风险较高，建议尽快检查。") if probability > THRESHOLD else \
                                    ("低风险", "风险较低，建议观察并定期复查。")
        else:
            prediction = model.predict(input_data)
            risk = prediction[0] * 100
            level = "未知风险"
            recommendation = "模型不支持概率预测，请检查模型类型。"

        return jsonify({'risk': round(risk, 2), 'level': level, 'recommendation': recommendation})

    except (ValueError, TypeError) as e:
        return jsonify({'error': '输入数据无效，请检查数据格式和范围。', 'details': str(e)})
    except Exception as e:
        return jsonify({'error': '处理请求时发生错误，请稍后重试。', 'details': str(e)})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info(f"应用正在运行，监听端口：{port}")
    app.run(host='0.0.0.0', port=port)