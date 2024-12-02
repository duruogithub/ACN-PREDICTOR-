from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)

# 定义阈值和模型路径
THRESHOLD = float(os.getenv("THRESHOLD", 0.136868298))
CURRENT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURRENT_DIR, "rf_model.pkl")

# 验证环境变量
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件未找到：{MODEL_PATH}")

# 加载模型
try:
    logger.info(f"正在加载模型文件：{MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict"):
        raise ValueError("加载的模型无效，请确认模型文件是否正确！")
except Exception as e:
    logger.error(f"加载模型时发生错误：{e}")
    raise RuntimeError(f"无法加载模型：{e}")

@app.route("/")
def index():
    """渲染主页"""
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"模板渲染错误：{e}")
        return f"模板渲染错误：{e}", 500

@app.route("/predict", methods=["POST"])
def predict():
    """处理预测请求"""
    try:
        # 提取并验证输入数据
        input_data = validate_input(request.form)
        
        # 执行模型预测
        prediction = make_prediction(input_data)
        
        return jsonify(prediction)
    except ValueError as ve:
        logger.warning(f"输入验证失败：{ve}")
        return jsonify({"error": "输入数据无效", "details": str(ve)}), 400
    except Exception as e:
        logger.error(f"预测处理失败：{e}")
        return jsonify({"error": "服务器错误", "details": str(e)}), 500

def validate_input(form):
    """验证并解析输入数据"""
    try:
        gender = int(form.get("gender", 0))
        age = int(form.get("age", 0))
        bmi = float(form.get("bmi", 0.0))
        residence = int(form.get("residence", 0))
        fx = int(form.get("fx", 0))
        bm = int(form.get("bm", 0))
        lwy = int(form.get("lwy", 0))
        smoke = int(form.get("smoke", 0))
        drink = int(form.get("drink", 0))
        fit = int(form.get("fit", 0))

        # 检查值是否在预期范围内
        if not (0 <= gender <= 1):
            raise ValueError("性别值必须为 0 或 1")
        if not (0 <= age <= 3):
            raise ValueError("年龄值必须在 0-3 之间")
        if not (0.0 <= bmi <= 50.0):
            raise ValueError("BMI 值必须在 0.0-50.0 之间")
        if not (0 <= residence <= 1):
            raise ValueError("居住地值必须为 0 或 1")
        if not all(0 <= v <= 1 for v in [fx, bm, lwy, smoke, drink, fit]):
            raise ValueError("二进制输入值必须为 0 或 1")

        return np.array([[gender, age, bmi, residence, fx, bm, lwy, smoke, drink, fit]])
    except ValueError as e:
        raise ValueError(f"输入数据验证失败：{e}")

def make_prediction(input_data):
    """使用模型进行预测并生成结果"""
    try:
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0, 1]
            risk = probability * 100
            if probability > 0.9:
                level, recommendation = "高风险", "风险非常高！建议立即检查。"
            elif probability > THRESHOLD:
                level, recommendation = "中等风险", "风险较高，建议尽快检查。"
            else:
                level, recommendation = "低风险", "风险较低，建议观察并定期复查。"
        else:
            prediction = model.predict(input_data)
            risk = prediction[0] * 100
            level, recommendation = "未知风险", "模型不支持概率预测，请检查模型类型。"

        return {"risk": round(risk, 2), "level": level, "recommendation": recommendation}
    except Exception as e:
        raise RuntimeError(f"预测失败：{e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"应用正在运行，监听端口：{port}")
    app.run(host="0.0.0.0", port=port)