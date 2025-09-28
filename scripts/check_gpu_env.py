#!/usr/bin/env python3
"""
检查GPU环境并提供特征筛选优化建议
"""
import sys
import os
import subprocess
import json

def check_gpu_env():
    """检查GPU环境配置"""
    print("=== GPU环境检查 ===\n")
    
    # 1. 检查CUDA
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ PyTorch CUDA可用")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("× PyTorch未安装")
    
    # 2. 检查XGBoost GPU支持
    xgb_gpu = False
    try:
        import xgboost as xgb
        # 尝试创建GPU booster
        try:
            dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
            bst = xgb.train({'tree_method': 'gpu_hist', 'gpu_id': 0}, dtrain, num_boost_round=1, verbose_eval=False)
            xgb_gpu = True
            print(f"✓ XGBoost GPU支持可用 (版本: {xgb.__version__})")
        except:
            print(f"× XGBoost已安装但GPU不可用 (版本: {xgb.__version__})")
    except ImportError:
        print("× XGBoost未安装")
    
    # 3. 检查CatBoost GPU支持
    cat_gpu = False
    try:
        import catboost
        print(f"✓ CatBoost已安装 (版本: {catboost.__version__})")
        # CatBoost GPU需要特殊构建版本
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(task_type='GPU', devices='0', iterations=1, verbose=False)
            cat_gpu = True
            print("  - GPU支持可用")
        except:
            print("  - GPU支持不可用（需要GPU版本）")
    except ImportError:
        print("× CatBoost未安装")
    
    # 4. 检查LightGBM GPU支持
    lgb_gpu = False
    try:
        import lightgbm as lgb
        # 检查是否编译了GPU支持
        if hasattr(lgb, 'LGBMClassifier'):
            try:
                model = lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1)
                lgb_gpu = True
                print(f"✓ LightGBM GPU支持可用 (版本: {lgb.__version__})")
            except:
                print(f"× LightGBM已安装但无GPU支持 (版本: {lgb.__version__})")
        else:
            print(f"× LightGBM已安装但版本过旧 (版本: {lgb.__version__})")
    except ImportError:
        print("× LightGBM未安装")
    
    # 5. Windows特定检查
    if sys.platform == 'win32':
        print("\n=== Windows环境优化建议 ===")
        print("✓ 检测到Windows系统")
        print("  - 建议使用n_jobs=4进行并行计算")
        print("  - 避免使用fork模式的多进程")
        print("  - 使用spawn或forkserver模式")
    
    # 6. 内存检查
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n=== 系统内存 ===")
        print(f"✓ 总内存: {memory.total / 1024**3:.1f} GB")
        print(f"✓ 可用内存: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        print("\n× psutil未安装，无法检查内存")
    
    # 7. 优化建议
    print("\n=== 优化建议 ===")
    
    if xgb_gpu:
        print("\n✓ XGBoost GPU加速可用，建议配置:")
        print("""
    xgb_params:
      tree_method: gpu_hist
      gpu_id: 0
      predictor: gpu_predictor
      max_bin: 256  # 适中的bin数量平衡速度和精度
      subsample: 0.8  # 避免过拟合
      colsample_bytree: 0.8
        """)
    else:
        print("\n× XGBoost GPU不可用，建议安装:")
        print("  pip install xgboost==2.0.3  # 确保CUDA版本匹配")
    
    if not cat_gpu:
        print("\n× CatBoost GPU不可用，建议安装:")
        print("  pip install catboost-gpu  # GPU版本")
    
    if not lgb_gpu:
        print("\n× LightGBM GPU不可用，建议从源码编译:")
        print("  参考: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html")
    
    print("\n=== 特征筛选性能优化策略 ===")
    print("""
1. Filter阶段优化:
   - 使用sample_frac=0.8采样计算IC/MI（不影响最终结果）
   - 设置batch_size=10000进行批量处理
   - Windows下使用n_jobs=4

2. Embedded阶段优化:
   - 优先使用XGBoost GPU (tree_backend: xgboost)
   - 保持n_estimators=300确保模型质量
   - 使用early_stopping避免过度训练

3. Permutation阶段优化:
   - 使用GPU加速的树模型
   - 保持repeats=5确保稳定性
   - 批量处理特征重要性计算

4. Wrapper阶段优化:
   - RFE使用batch_eval=true批量评估
   - GA使用parallel_eval=true并行评估种群
   - 保持较大的种群规模确保搜索质量

5. 数据处理优化:
   - 使用float32代替float64（GPU计算更快）
   - 提前计算并缓存特征工程结果
   - 使用内存映射处理大数据集
    """)
    
    return {
        'cuda': cuda_available,
        'xgboost_gpu': xgb_gpu,
        'catboost_gpu': cat_gpu,
        'lightgbm_gpu': lgb_gpu,
        'platform': sys.platform
    }

if __name__ == "__main__":
    env_status = check_gpu_env()
    
    # 生成优化配置建议
    if env_status['xgboost_gpu']:
        print("\n=== 推荐的特征筛选启动命令 ===")
        print("python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --gpu --n_jobs 4")
    else:
        print("\n=== 当前环境下的启动命令 ===")
        print("python -m pipelines.run_feature_screening --config pipelines/configs/feature_selection.yaml --n_jobs 4")
