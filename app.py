from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 定义ROC曲线的最佳阈值
THRESHOLD = 0.136868298  # 根据ROC曲线计算得出的最佳阈值

# 加载模型
MODEL_PATH = 'rf_model.pkl'
try:
    model = joblib.load(MODEL_PATH)  # 使用 joblib 加载模型
    if not hasattr(model, 'predict'):
        raise ValueError("加载的模型无效，请确保模型文件正确！")
except FileNotFoundError:
    raise FileNotFoundError(f"模型文件未找到，请确认路径：{MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"加载模型时发生错误: {e}")

@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理前端请求并返回预测结果"""
    try:
        # 提取表单数据并转换为数值类型
        gender = int(request.form.get('gender'))
        age = int(request.form.get('age'))
        bmi = float(request.form.get('bmi'))  # BMI 应为浮点型
        residence = int(request.form.get('residence'))
        fx = int(request.form.get('fx'))
        bm = int(request.form.get('bm'))
        lwy = int(request.form.get('lwy'))
        smoke = int(request.form.get('smoke'))
        drink = int(request.form.get('drink'))
        fit = int(request.form.get('fit'))

        # 构造输入数据
        input_data = np.array([[gender, age, bmi, residence, fx, bm, lwy, smoke, drink, fit]])

        # 预测风险
        if hasattr(model, 'predict_proba'):
            # 获取类别 "1" 的概率
            probability = model.predict_proba(input_data)[0, 1]
            risk = probability * 100  # 转换为百分比形式

            # 根据风险概率划分建议
            if probability > 0.9:
                recommendation = "风险非常高！建议立即进行肠镜检查，并咨询医生。"
            elif probability > THRESHOLD:
                recommendation = "风险较高，建议尽快进行肠镜检查。"
            else:
                recommendation = "风险较低，建议观察并定期复查。"

        else:
            # 如果模型不支持预测概率，直接使用分类结果
            prediction = model.predict(input_data)
            risk = prediction[0] * 100  # 假设直接用分类结果
            recommendation = "模型不支持概率预测，请检查模型类型。"

        # 返回 JSON 格式的预测结果
        return jsonify({
            'risk': round(risk, 2),  # 风险概率（百分比）
            'level': "高风险" if probability > THRESHOLD else "低风险",  # 风险级别
            'recommendation': recommendation  # 具体建议
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 以调试模式运行应用
    app.run(debug=True)