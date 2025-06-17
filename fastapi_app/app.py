from fastapi import FastAPI, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import subprocess
import os
import json
from typing import Dict, Any
import re
import uuid

app = FastAPI()

# 用于存储任务状态和结果
task_status: Dict[str, Dict[str, Any]] = {}

# 定义主项目目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = os.path.join(PROJECT_DIR, "main.py")

print("Main script path:", MAIN_SCRIPT)  # 调试：打印 main.py 的路径

# 根路径返回表单页面
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>DCIMVC 训练平台</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                form { max-width: 600px; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
                button { background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #2980b9; }
            </style>
        </head>
        <body>
            <h1>DCIMVC 多视图聚类训练参数</h1>
            <form action="/run_training" method="post">
                <div class="form-group">
                    <label>缺失率 (0.1-0.9):</label>
                    <input type="number" name="missing_rate" step="0.1" min="0.1" max="0.9" value="0.3" required>
                </div>
                
                <div class="form-group">
                    <label>数据集编号:</label>
                    <select name="dataset" required>
                        <option value="0">0: Caltech7-5V</option>
                        <option value="1">1: Multi-Fashion</option>
                        <option value="2" selected>2: UCI_Digits</option>
                        <option value="3">3: HandWritten</option>
                        <option value="4">4: MNIST_USPS</option>
                        <option value="5">5: NoisyMNIST</option>
                        <option value="8">8: Scene-15</option>
                        <option value="9">9: LandUse-21</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>训练轮数:</label>
                    <input type="number" name="epochs" value="200">
                </div>
                
                <div class="form-group">
                    <label>初始化训练轮数:</label>
                    <input type="number" name="initial_epochs" value="200">
                </div>
                
                <div class="form-group">
                    <label>批量大小:</label>
                    <input type="number" name="batch_size" value="256">
                </div>
                
                <div class="form-group">
                    <label>学习率:</label>
                    <input type="number" name="learning_rate" step="0.0001" value="0.0005">
                </div>
                
                <div class="form-group">
                    <label>先验学习率:</label>
                    <input type="number" name="prior_learning_rate" step="0.01" value="0.05">
                </div>
                
                <div class="form-group">
                    <label>潜在空间维度:</label>
                    <input type="number" name="z_dim" value="10">
                </div>
                
                <div class="form-group">
                    <label>学习率衰减步长:</label>
                    <input type="number" name="lr_decay_step" value="10">
                </div>
                
                <div class="form-group">
                    <label>学习率衰减因子:</label>
                    <input type="number" name="lr_decay_factor" step="0.01" value="0.9">
                </div>
                
                <div class="form-group">
                    <label>日志打印间隔:</label>
                    <input type="number" name="interval" value="20">
                </div>
                
                <div class="form-group">
                    <label>测试次数:</label>
                    <input type="number" name="test_times" value="2">
                </div>
                
                <div class="form-group">
                    <label>超参数 alpha:</label>
                    <input type="number" name="alpha" step="0.1" value="5">
                </div>
                
                <div class="form-group">
                    <label>超参数 lamb0:</label>
                    <input type="number" name="lamb0" step="0.1" value="1">
                </div>
                
                <div class="form-group">
                    <label>超参数 lamb1:</label>
                    <input type="number" name="lamb1" step="0.1" value="1">
                </div>
                
                <div class="form-group">
                    <label>超参数 lamb2:</label>
                    <input type="number" name="lamb2" step="0.1" value="1">
                </div>
                
                <div class="form-group">
                    <label>超参数 lamb3:</label>
                    <input type="number" name="lamb3" step="0.1" value="1">
                </div>
                
                <div class="form-group">
                    <label>随机种子0:</label>
                    <input type="number" name="seed0" value="1">
                </div>
                
                <div class="form-group">
                    <label>随机种子1:</label>
                    <input type="number" name="seed1" value="1">
                </div>
                
                <button type="submit">开始训练</button>
            </form>
        </body>
    </html>
    """

def run_main_script(missing_rate: float, dataset: int, epochs: int = 200, initial_epochs: int = 200, batch_size: int = 256, 
                    learning_rate: float = 0.0005, prior_learning_rate: float = 0.05, z_dim: int = 10, lr_decay_step: int = 10,
                    lr_decay_factor: float = 0.9, interval: int = 20, test_times: int = 10, alpha: float = 5, 
                    lamb0: float = 1, lamb1: float = 1, lamb2: float = 1, lamb3: float = 1, seed0: int = 1, 
                    seed1: int = 1) -> str:
    """
    调用 main.py 脚本执行训练和评估。

    Args:
        missing_rate (float): 缺失率.
        dataset (int): 数据集编号.
        epochs (int): 训练 epoch 数.
        initial_epochs (int): 初始化训练 epoch 数.
        batch_size (int): 批量大小.
        learning_rate (float): 学习率.
        prior_learning_rate (float): 先验学习率.
        z_dim (int): 潜在空间维度.
        lr_decay_step (int): 学习率衰减步长.
        lr_decay_factor (float): 学习率衰减因子.
        interval (int): 日志打印间隔.
        test_times (int): 测试次数.
        alpha (float): 超参数 alpha.
        lamb0 (float): 超参数 lamb0.
        lamb1 (float): 超参数 lamb1.
        lamb2 (float): 超参数 lamb2.
        lamb3 (float): 超参数 lamb3.
        seed0 (int): 随机种子0.
        seed1 (int): 随机种子1.

    Returns:
        str: 命令执行结果.
    """
    try:
        # 构建命令
        command = [
            "python", MAIN_SCRIPT,
            "--missing_rate", str(missing_rate),
            "--dataset", str(dataset),
            "--epochs", str(epochs),
            "--initial_epochs", str(initial_epochs),
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--prior_learning_rate", str(prior_learning_rate),
            "--z_dim", str(z_dim),
            "--lr_decay_step", str(lr_decay_step),
            "--lr_decay_factor", str(lr_decay_factor),
            "--interval", str(interval),
            "--test_times", str(test_times),
            "--alpha", str(alpha),
            "--lamb0", str(lamb0),
            "--lamb1", str(lamb1),
            "--lamb2", str(lamb2),
            "--lamb3", str(lamb3),
            "--seed0", str(seed0),
            "--seed1", str(seed1),
        ]
        # 执行命令，并切换工作目录到 main.py 所在目录
        result = subprocess.run(command, capture_output=True, text=True, cwd=os.path.dirname(MAIN_SCRIPT))
        return result.stdout
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing main.py: {str(e)}")

@app.post("/run_training")
async def run_training(
    background_tasks: BackgroundTasks,
    request: Request,
    missing_rate: float = Form(...),
    dataset: int = Form(...),
    epochs: int = Form(default=200),
    initial_epochs: int = Form(default=200),
    batch_size: int = Form(default=256),
    learning_rate: float = Form(default=0.0005),
    prior_learning_rate: float = Form(default=0.05),
    z_dim: int = Form(default=10),
    lr_decay_step: int = Form(default=10),
    lr_decay_factor: float = Form(default=0.9),
    interval: int = Form(default=20),
    test_times: int = Form(default=10),
    alpha: float = Form(default=5),
    lamb0: float = Form(default=1),
    lamb1: float = Form(default=1),
    lamb2: float = Form(default=1),
    lamb3: float = Form(default=1),
    seed0: int = Form(default=1),
    seed1: int = Form(default=1)
):
    """
    API 路径：/run_training
    方法：POST

    调用 DCIMVC 的主脚本进行训练和测试。

    参数：
        missing_rate (float): 数据缺失率.
        dataset (int): 数据集编号.
        epochs (int): 训练 epoch 数.
        initial_epochs (int): 初始化训练 epoch 数.
        batch_size (int): 批量大小.
        learning_rate (float): 学习率.
        prior_learning_rate (float): 先验学习率.
        z_dim (int): 潜在空间维度.
        lr_decay_step (int): 学习率衰减步长.
        lr_decay_factor (float): 学习率衰减因子.
        interval (int): 日志打印间隔.
        test_times (int): 测试次数.
        alpha (float): 超参数 alpha.
        lamb0 (float): 超参数 lamb0.
        lamb1 (float): 超参数 lamb1.
        lamb2 (float): 超参数 lamb2.
        lamb3 (float): 超参数 lamb3.
        seed0 (int): 随机种子0.
        seed1 (int): 随机种子1.

    返回：
        根据请求头Accept返回HTML或JSON格式的响应
    """
    try:
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        # 初始化任务状态
        task_status[task_id] = {"status": "running", "result": None}

        # 创建后台任务
        background_tasks.add_task(
            run_training_task,
            task_id,
            missing_rate,
            dataset,
            epochs,
            initial_epochs,
            batch_size,
            learning_rate,
            prior_learning_rate,
            z_dim,
            lr_decay_step,
            lr_decay_factor,
            interval,
            test_times,
            alpha,
            lamb0,
            lamb1,
            lamb2,
            lamb3,
            seed0,
            seed1
        )
        
        # 返回提交页面
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>训练任务提交</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #27ae60; }}
                    .message {{ 
                        background-color: #f8f9fa; 
                        padding: 20px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-top: 20px;
                    }}
                    .back-btn {{ 
                        display: inline-block; 
                        margin-top: 20px; 
                        padding: 10px 20px;
                        background-color: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                    }}
                    .back-btn:hover {{ background-color: #2980b9; }}
                </style>
                <script>
                    // 定期刷新页面以检查任务状态
                    function checkTaskStatus() {{
                        fetch('/run_training/status/{task_id}')
                            .then(response => response.json())
                            .then(data => {{
                                if (data.status === 'completed') {{
                                    window.location.href = '/run_training/result/{task_id}';
                                }} else if (data.status === 'error') {{
                                    alert('任务出错: ' + data.error);
                                    window.location.href = '/';
                                }}
                            }});
                    }}
                    // 每 5 秒检查一次任务状态
                    setInterval(checkTaskStatus, 5000);
                </script>
            </head>
            <body>
                <h1>训练任务已提交</h1>
                <div class="message">
                    <p>训练任务正在后台处理，请稍候...</p>
                </div>
                <a href="/" class="back-btn">返回参数设置</a>
            </body>
        </html>
        """)
            
    except HTTPException as e:
        raise
    except Exception as e:
        # 错误处理也支持HTML格式
        accept = request.headers.get("accept", "")
        error_detail = f"训练失败: {str(e)}"
        if "text/html" in accept:
            return HTMLResponse(content=f"<h1>错误</h1><p>{error_detail}</p>", status_code=500)
        else:
            raise HTTPException(status_code=500, detail=error_detail)

async def run_training_task(
    task_id: str,
    missing_rate: float,
    dataset: int,
    epochs: int,
    initial_epochs: int,
    batch_size: int,
    learning_rate: float,
    prior_learning_rate: float,
    z_dim: int,
    lr_decay_step: int,
    lr_decay_factor: float,
    interval: int,
    test_times: int,
    alpha: float,
    lamb0: float,
    lamb1: float,
    lamb2: float,
    lamb3: float,
    seed0: int,
    seed1: int
):
    try:
        output = run_main_script(
            missing_rate=missing_rate,
            dataset=dataset,
            epochs=epochs,
            initial_epochs=initial_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            prior_learning_rate=prior_learning_rate,
            z_dim=z_dim,
            lr_decay_step=lr_decay_step,
            lr_decay_factor=lr_decay_factor,
            interval=interval,
            test_times=test_times,
            alpha=alpha,
            lamb0=lamb0,
            lamb1=lamb1,
            lamb2=lamb2,
            lamb3=lamb3,
            seed0=seed0,
            seed1=seed1
        )
        print("Main.py output:")  # 调试：打印 main.py 的输出
        print(output)
        # 处理解析逻辑
        result = {
            "ACC": None,
            "NMI": None,
            "ARI": None,
            "PUR": None
        }
        final_results_found = False
        # 使用正则表达式匹配最终结果行
        pattern = r"FINAL_RESULTS:\s+Average ACC\s+(\d+\.\d+)\s+Average NMI\s+(\d+\.\d+)\s+Average ARI\s+(\d+\.\d+)\s+Average PUR\s+(\d+\.\d+)"
        
        for line in output.splitlines():
            stripped_line = line.strip()
            match = re.search(pattern, stripped_line)
            if match:
                result["ACC"] = float(match.group(1))
                result["NMI"] = float(match.group(2))
                result["ARI"] = float(match.group(3))
                result["PUR"] = float(match.group(4))
                final_results_found = True
                break
        
        if final_results_found:
            print(f"训练结果：ACC={result['ACC']}, NMI={result['NMI']}, ARI={result['ARI']}, PUR={result['PUR']}")
            task_status[task_id] = {
                "status": "completed",
                "result": result
            }
        else:
            print("未找到最终结果")
            task_status[task_id] = {
                "status": "error",
                "error": "未找到最终结果"
            }
    except Exception as e:
        print(f"训练任务出错：{str(e)}")
        task_status[task_id] = {
            "status": "error",
            "error": str(e)
        }

@app.get("/run_training/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id in task_status:
        return task_status[task_id]
    else:
        return {"status": "unknown", "error": "任务ID不存在"}

@app.get("/run_training/result/{task_id}")
async def get_task_result(task_id: str):
    if task_id in task_status and task_status[task_id].get("status") == "completed":
        result = task_status[task_id]["result"]
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>训练结果</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #27ae60; }}
                    .result-container {{ 
                        background-color: #f8f9fa; 
                        padding: 20px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-top: 20px;
                    }}
                    .metric {{ 
                        font-size: 1.2em; 
                        margin: 10px 0; 
                        padding: 10px;
                        background-color: white;
                        border-left: 4px solid #3498db;
                    }}
                    .metric-value {{ font-weight: bold; color: #2980b9; }}
                    .back-btn {{ 
                        display: inline-block; 
                        margin-top: 20px; 
                        padding: 10px 20px;
                        background-color: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                    }}
                    .back-btn:hover {{ background-color: #2980b9; }}
                </style>
            </head>
            <body>
                <h1>训练完成！</h1>
                <div class="result-container">
                    <div class="metric">ACC: <span class="metric-value">{result["ACC"]*100:.2f}%</span></div>
                    <div class="metric">NMI: <span class="metric-value">{result["NMI"]*100:.2f}%</span></div>
                    <div class="metric">ARI: <span class="metric-value">{result["ARI"]*100:.2f}%</span></div>
                    <div class="metric">PUR: <span class="metric-value">{result["PUR"]*100:.2f}%</span></div>
                </div>
                <a href="/" class="back-btn">返回参数设置</a>
            </body>
        </html>
        """)
    elif task_id in task_status and task_status[task_id].get("status") == "error":
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>训练错误</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #e74c3c; }}
                    .error-container {{ 
                        background-color: #fdeaea; 
                        padding: 20px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-top: 20px;
                    }}
                    .back-btn {{ 
                        display: inline-block; 
                        margin-top: 20px; 
                        padding: 10px 20px;
                        background-color: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                    }}
                    .back-btn:hover {{ background-color: #2980b9; }}
                </style>
            </head>
            <body>
                <h1>训练任务出错</h1>
                <div class="error-container">
                    <p>{task_status[task_id].get("error", "未知错误")}</p>
                </div>
                <a href="/" class="back-btn">返回参数设置</a>
            </body>
        </html>
        """)
    else:
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>训练任务提交</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #27ae60; }}
                    .message {{ 
                        background-color: #f8f9fa; 
                        padding: 20px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-top: 20px;
                    }}
                    .back-btn {{ 
                        display: inline-block; 
                        margin-top: 20px; 
                        padding: 10px 20px;
                        background-color: #3498db;
                        color: white;
                        text-decoration: none;
                        border-radius: 4px;
                    }}
                    .back-btn:hover {{ background-color: #2980b9; }}
                </style>
                <script>
                    // 定期刷新页面以检查任务状态
                    function checkTaskStatus() {{
                        fetch('/run_training/status/{task_id}')
                            .then(response => response.json())
                            .then(data => {{
                                if (data.status === 'completed') {{
                                    window.location.href = '/run_training/result/{task_id}';
                                }} else if (data.status === 'error') {{
                                    window.location.href = '/run_training/result/{task_id}';
                                }}
                            }});
                    }}
                    // 每 5 秒检查一次任务状态
                    setInterval(checkTaskStatus, 5000);
                </script>
            </head>
            <body>
                <h1>训练任务已提交</h1>
                <div class="message">
                    <p>训练任务正在后台处理，请稍候...</p>
                </div>
                <a href="/" class="back-btn">返回参数设置</a>
            </body>
        </html>
        """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)