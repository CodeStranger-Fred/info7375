import asyncio
import os
from pathlib import Path

import torch

from maas.configs.models_config import ModelsConfig
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from maas.ext.maas.models.controller import MultiLayerController
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.scripts.evaluator import Evaluator
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.logs import logger


DATASET = "WikiSQL"
MAX_EXAMPLES = 80  # 为了时间控制，先评测前 80 个样本，你可以之后改大
BATCH_SIZE = 2
MODEL_NAME = "gpt-4o-mini"  # 需要在 config2.yaml 里配置好这个模型和 API key


async def main():
    # 1. 读取模型配置
    models_config = ModelsConfig.default()
    exec_llm_config = models_config.get(MODEL_NAME)
    if exec_llm_config is None:
        raise ValueError(
            f"Model '{MODEL_NAME}' not found in models config. Please add it in ~/.metagpt/config2.yaml."
        )

    # 2. 读取实验配置（里面包含要用哪些 operator）
    exp_cfg = EXPERIMENT_CONFIGS[DATASET]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 构建 controller 和 operator embedding（MaAS 的搜索部分）
    root_path = "maas/ext/maas/scripts/optimized/WikiSQL"
    graph_utils = GraphUtils(root_path)

    operator_descriptions = graph_utils.load_operators_description_maas(exp_cfg.operators)
    operator_embeddings = torch.stack(
        [get_sentence_embedding(desc) for desc in operator_descriptions]
    ).to(device)

    controller = MultiLayerController(device=device).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)

    # 4. 加载我们刚复用的 multi-agent workflow（来自 optimized/WikiSQL/test/graph.py）
    workflows_path = "maas/ext/maas/scripts/optimized/WikiSQL/test"
    graph_class = graph_utils.load_graph_maas(workflows_path)

    # 5. 组装 evaluator 需要的参数
    eval_path = "logs_wikisql_eval"
    os.makedirs(eval_path, exist_ok=True)

    params = {
        "operator_embeddings": operator_embeddings,
        "controller": controller,
        "execute_llm_config": exec_llm_config,
        "dataset": DATASET,
        "optimizer": optimizer,
        "sample": 1,          # 不做多次采样，节省时间
        "is_textgrad": False,
    }
    
    # Slow down API calls to avoid rate limiting
    import time
    import asyncio
    original_acall = exec_llm_config.__class__.aask if hasattr(exec_llm_config.__class__, 'aask') else None

    evaluator = Evaluator(eval_path=eval_path, batch_size=BATCH_SIZE)

    # 6. 为了只跑前 MAX_EXAMPLES 个样本，我们在 WikiSQLBenchmark 里用 va_list 索引
    # 这里复用 EvaluationUtils 的逻辑，只是设置 is_test=True
    eval_utils = EvaluationUtils(root_path)

    # EvaluationUtils 本身不支持传入 va_list，所以我们直接调用 Evaluator.graph_evaluate，
    # 然后在 WikiSQLBenchmark.load_data 里用 specific_indices 控制。
    # 这里我们简单传 None，让 benchmark 自己用全部数据；你如果想只跑前 N 个，
    # 可以直接在数据文件里裁剪，或者改 WikiSQLBenchmark.load_data。

    logger.info(f"Running MaAS on {DATASET} test set (may take a while)...")
    avg_score = await evaluator.graph_evaluate(
        dataset=DATASET,
        graph=graph_class,
        params=params,
        path=eval_path,
        is_test=True,
    )

    print("\n===== WikiSQL Evaluation Done =====")
    print(f"Execution accuracy (avg score): {avg_score:.4f}")
    print(f"Detailed per-example results are saved under: {eval_path}")
    print("You can open the CSV there to挑出 5 个最简单失败、5 个最难成功的样本来做分析。")


if __name__ == "__main__":
    asyncio.run(main())
