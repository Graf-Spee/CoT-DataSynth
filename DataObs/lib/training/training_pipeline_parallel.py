"""
Training pipeline management
Handles training execution, GPU allocation, and result collection
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch

logger = logging.getLogger(__name__)


# 在 __init__ 或首次调用时初始化 NVML 一次即可，这里封装为静态工具
def _query_gpu_free_mb(gpu_id: int) -> int:
    """查询指定 GPU 剩余显存 (MB) """
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.free // 1024 ** 2


def _query_pool_free_mb(gpu_pool: List[int]) -> Dict[int, int]:
    """批量查询 GPU 池剩余显存"""
    return {gid: _query_gpu_free_mb(gid) for gid in gpu_pool}


class TrainingPipelineParallel:
    """Manage training execution for multiple splits"""

    def __init__(self, output_dir: str, cot_datasynth_dir: str):
        """
        Initialize training pipeline

        Args:
            output_dir: Directory to save training results
            cot_datasynth_dir: Path to CoT-DataSynth project directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cot_datasynth_dir = Path(cot_datasynth_dir)
        self.training_dir = self.output_dir / "training"
        self.training_dir.mkdir(exist_ok=True)
        self.eval_dir = self.output_dir / "eval"
        self.eval_dir.mkdir(exist_ok=True)
        self.config_file = self.output_dir / "training_configs.json"
        self.log_file = self.training_dir / "training_log.json"
        logger.info(f"TrainingPipeline initialized: {self.output_dir}")

    def prepare_training_configs(
        self,
        splits_dir: str,
        base_config: Dict[str, Any],
        n_splits: int,
        base_model_id: str,
        estimated_vram_mb: int = 30000,   # 每个 split 预估显存（MB）
    ) -> List[Dict[str, Any]]:
        configs = []
        for split_id in range(n_splits):
            split_data_path = Path(splits_dir) / f"split_{split_id}.parquet"
            if not split_data_path.exists():
                logger.warning(f"Split data not found: {split_data_path}")
                continue

            split_output_dir = self.training_dir / f"split_{split_id}"
            split_output_dir.mkdir(exist_ok=True)
            eval_output_dir = self.eval_dir / f"split_{split_id}"
            eval_output_dir.mkdir(exist_ok=True)

            config = {
                'split_id': split_id,
                'data_path': str(split_data_path),
                'output_dir': str(split_output_dir),
                'eval_output_dir': str(eval_output_dir),
                'gpu_ids': [],                       # 运行时由调度器分配单卡
                'estimated_vram_mb': estimated_vram_mb,
                'base_model_id': base_model_id,
                **base_config
            }
            configs.append(config)

        # Save configs
        with open(self.config_file, 'w') as f:
            json.dump(configs, f, indent=2)

        logger.info(f"Prepared {len(configs)} training configurations")
        return configs

    def run_training(
        self,
        config: Dict[str, Any],
        script_path: str,
        val_data_path: Optional[str] = '/data/open_datasets/GSM8K/main/test-00000-of-00001.parquet',
        num_epochs: int = 15,
        log_file: Optional[Path] = None,       # 新增：重定向日志
    ) -> Optional[subprocess.Popen]:
        """
        执行单份训练。blocking=False 时返回 Popen 对象，由上层调度器管理生命周期。
        """
        split_id = config['split_id']
        gpu_ids = config['gpu_ids']
        data_path = config['data_path']
        output_dir = config['output_dir']
        base_model_id = config['base_model_id']

        # 单卡字符串（调度器保证只分配一张卡）
        gpu_str = str(gpu_ids[0]) if gpu_ids else '0'

        script_path = Path(script_path)
        if not script_path.is_absolute():
            script_path = self.cot_datasynth_dir / script_path

        cmd = [
            'bash', str(script_path),
            str(base_model_id),
            str(data_path),
            val_data_path,
            str(output_dir),
            gpu_str,
            str(num_epochs),
        ]
        # Add extra config parameters
        for key, value in config.items():
            if key not in ['split_id', 'gpu_ids', 'data_path', 'output_dir',
                           'eval_output_dir', 'base_model_id', 'estimated_vram_mb']:
                cmd.append(f'{key}={value}')

        logger.info(f"Launch training for split {split_id} on GPU {gpu_str}: {' '.join(cmd)}")

        try:
            # None-Blocking: 日志写入文件，避免 PIPE 阻塞子进程
            if log_file is None:
                log_file = self.training_dir / f"split_{split_id}" / "training.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fout = open(log_file, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.cot_datasynth_dir),
                stdout=fout,
                stderr=subprocess.STDOUT,
            )
            # 将文件句柄绑定到 proc，便于后续关闭
            proc._log_fout = fout
            return proc

        except subprocess.TimeoutExpired:
            logger.error(f"Training for split {split_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Training for split {split_id} exception: {e}")
            return None
    
    def run_all_trainings(
        self,
        configs: List[Dict[str, Any]],
        script_path: str,
        skip_completed: bool = True,
        val_data_path: Optional[str] = None,
        eval_script_path: Optional[str] = None,
        eval_data_name: Optional[str] = None,
        gpu_pool: Optional[List[int]] = None,  # 可用 GPU 列表，如 [0,1,2,3]
        safety_margin_mb: int = 2048,          # 安全余量
        poll_interval: int = 5,                # 显存轮询间隔（秒）
        num_epochs: int = 15,
    ) -> Dict[int, bool]:
        """
        并行调度多 split 训练。每个 split 独占一张卡，程序自动判断剩余显存是否足够。
        """
        results: Dict[int, bool] = {}

        # ---------- 并行调度 ----------
        nvmlInit()
        try:
            pool = gpu_pool or list(range(torch.cuda.device_count())) if hasattr(torch.cuda, 'device_count') else [0]

            # 初始化 pending 队列
            pending: List[Dict[str, Any]] = []
            for config in configs:
                sid = config['split_id']
                if skip_completed and self.is_split_completed(sid):
                    logger.info(f"Split {sid} already completed, skip")
                    results[sid] = True
                    continue
                pending.append(config)

            # running: List[Tuple[Popen, config, log_fout]]
            running: List[Tuple[subprocess.Popen, Dict[str, Any]]] = []

            while pending or running:
                # ---- 1. 回收已结束的训练 ----
                finished = []
                for proc, config in running:
                    ret = proc.poll()
                    if ret is not None:
                        finished.append((proc, config))
                        sid = config['split_id']
                        success = (ret == 0)
                        results[sid] = success

                        # 关闭日志文件句柄
                        if hasattr(proc, '_log_fout'):
                            proc._log_fout.close()

                        if success:
                            # 非阻塞模式下，日志已写入文件，这里可再读取解析
                            log_file = self.training_dir / f"split_{sid}" / "training.log"
                            if log_file.exists():
                                stdout = log_file.read_text()
                                metrics = self._parse_training_metrics(stdout, "")
                                self._save_training_results(sid, config['output_dir'], metrics)
                            self._update_split_log(sid, "completed", "Training succeeded")
                        else:
                            self._update_split_log(sid, "failed", f"Exit code {ret}")
                            logger.error(f"Split {sid} training failed")

                for item in finished:
                    running.remove(item)

                # ---- 2. 尝试为 pending 任务分配 GPU 并启动 ----
                if pending:
                    free_mem = _query_pool_free_mb(pool)

                    # 预扣机制：减去此轮 dispatch 的新任务所占用的预估显存，防止超卖
                    reserved = {gid: 0 for gid in pool}

                    next_pending: List[Dict[str, Any]] = []
                    for config in pending:
                        need = config.get('estimated_vram_mb', 8000) + safety_margin_mb
                        assigned = None

                        # 遍历 GPU 池，找第一张"实际剩余 - 已预留"够用的卡
                        for gid in pool:
                            if free_mem[gid] - reserved[gid] >= need:
                                assigned = gid
                                reserved[gid] += config.get('estimated_vram_mb', 8000)
                                break

                        if assigned is not None:
                            config['gpu_ids'] = [assigned]
                            self._update_split_log(config['split_id'], "running")
                            proc = self.run_training(config, script_path,
                                                     val_data_path,
                                                     num_epochs=num_epochs)
                            if proc:
                                running.append((proc, config))
                                logger.info(f"Start split {config['split_id']} on GPU {assigned}")
                            else:
                                # 启动失败，记为失败不再重试
                                results[config['split_id']] = False
                        else:
                            # 显存不足，保留在 pending
                            next_pending.append(config)

                    pending = next_pending

                # ---- 3. 若仍有 pending，等待后重试 ----
                if pending:
                    logger.info(
                        f"Waiting: {len(pending)} pending, {len(running)} running | "
                        f"GPU free: { {k: _query_gpu_free_mb(k) for k in pool} }"
                    )
                    time.sleep(poll_interval)

            # ---------- 全部训练结束后，顺序执行评估 ----------
            if eval_script_path and eval_data_name:
                for config in configs:
                    sid = config['split_id']
                    if results.get(sid):
                        # eval 复用该 split 被分配过的那张卡（训练已结束，显存已释放）
                        if not config.get('gpu_ids'):
                            config['gpu_ids'] = [pool[0]]
                        self.run_evaluation(config, eval_script_path, eval_data_name)

        finally:
            nvmlShutdown()

        logger.info(f"All done: {sum(results.values())}/{len(results)} succeeded")
        return results

    def run_evaluation(
        self,
        config: Dict[str, Any],
        eval_script_path: str,
        eval_data_name: str,
    ) -> bool:
        """
        Run evaluation on a trained split

        Args:
            config: Training configuration
            eval_script_path: Path to evaluation script

        Returns:
            True if evaluation succeeded, False otherwise
        """
        gpu_ids = config['gpu_ids']
        output_dir = config['output_dir']
        eval_output_dir = config['eval_output_dir']
        base_model_id = config['base_model_id']

        # 查找最新的 checkpoint (global_step_*)
        output_path = Path(output_dir)
        checkpoints = list(output_path.glob("global_step_*"))
        if not checkpoints:
            logger.warning(f"No checkpoints found in {output_dir}")
            return False

        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split('_')[2]))[-1]
        logger.info(f"Using checkpoint: {latest_checkpoint}")

        # Prepare GPU string
        gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else '0'
        
        # Prepare command
        eval_script_path = Path(eval_script_path)
        if not eval_script_path.is_absolute():
            eval_script_path = self.cot_datasynth_dir / eval_script_path

        cmd = [
            'bash',
            str(eval_script_path),
            str(latest_checkpoint),     # checkpoint path (param 1)
            str(base_model_id),         # base model (param 2)
            str(eval_data_name),        # dataset name (param 3)
            str(eval_output_dir),       # eval output dir (param 4)
            str(gpu_str),               # gpu_id (param 5)
        ]

        logger.info(f"Running evaluation: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.cot_datasynth_dir}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.cot_datasynth_dir),
                timeout=3600,  # 1 hour timeout for evaluation
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # 尝试从评估输出中提取准确率
                with open(f'{eval_output_dir}/logs/evaluation.log', 'r', encoding='utf-8') as f:
                    output = f.read()
                import re

                # 尝试多种模式匹配准确率
                acc_patterns = [
                    r'accuracy[:\s]+([0-9.]+)',
                    r'test_score[:\s]+([0-9.]+)',
                    r'pass@1[:\s]+([0-9.]+)',
                ]

                accuracy = None
                for pattern in acc_patterns:
                    acc_match = re.search(pattern, output, re.IGNORECASE)
                    if acc_match:
                        accuracy = float(acc_match.group(1))
                        logger.info(f"Evaluation accuracy: {accuracy}")
                        break

                if accuracy is not None:
                    # 保存到 training_results.json
                    results_file = Path(output_dir) / "training_results.json"
                    if results_file.exists():
                        with open(results_file) as f:
                            results = json.load(f)
                    else:
                        results = {}

                    results['test_accuracy'] = accuracy
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"Saved test_accuracy to {results_file}")
                else:
                    logger.warning("Could not extract accuracy from evaluation output")

                return True
            else:
                logger.error(f"Evaluation failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout[-1000:]}")  # 最后 1000 字符
                logger.error(f"STDERR: {result.stderr[-1000:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation timed out after 3600 seconds")
            return False
        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _parse_training_metrics(self, stdout: str, stderr: str) -> Dict[str, float]:
        """从训练输出中解析 metrics"""
        metrics = {}
        output = stdout + stderr

        # 尝试匹配常见的 metric 模式
        import re

        # 匹配 "train_loss: 0.123" 或 "train/loss: 0.123"
        loss_patterns = [
            r'train[/_]loss[:\s]+([0-9.]+)',
            r'loss[:\s]+([0-9.]+)',
            r'Final.*loss[:\s]+([0-9.]+)',
        ]
        for pattern in loss_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['train_loss'] = float(match.group(1))
                break

        # 匹配验证准确率
        acc_patterns = [
            r'val[/_]acc(?:uracy)?[:\s]+([0-9.]+)',
            r'accuracy[:\s]+([0-9.]+)',
            r'eval[/_]acc(?:uracy)?[:\s]+([0-9.]+)',
            r'Final.*acc(?:uracy)?[:\s]+([0-9.]+)',
        ]
        for pattern in acc_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['val_accuracy'] = float(match.group(1))
                break

        # 匹配验证 loss
        val_loss_patterns = [
            r'val[/_]loss[:\s]+([0-9.]+)',
            r'eval[/_]loss[:\s]+([0-9.]+)',
        ]
        for pattern in val_loss_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['val_loss'] = float(match.group(1))
                break

        return metrics

    def _save_training_results(self, split_id: int, output_dir: str, metrics: Dict[str, float]):
        """保存训练结果到 JSON 文件"""
        output_path = Path(output_dir) / "training_results.json"

        # 如果文件已存在，先读取
        existing = {}
        if output_path.exists():
            try:
                with open(output_path) as f:
                    existing = json.load(f)
            except:
                pass

        # 合并 metrics
        existing.update(metrics)
        existing['split_id'] = split_id
        existing['status'] = 'completed'
        existing['timestamp'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved training metrics to {output_path}")

    def collect_training_results(
        self,
        n_splits: int,
        metric_keys: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Collect training results from output directories

        Args:
            n_splits: Number of splits
            metric_keys: Keys to extract from results (e.g., ['accuracy', 'loss'])

        Returns:
            Dictionary mapping split_id to metrics
        """
        results = {}

        for split_id in range(n_splits):
            split_output_dir = self.training_dir / f"split_{split_id}"
            results_file = split_output_dir / "training_results.json"

            if results_file.exists():
                try:
                    with open(results_file) as f:
                        split_results = json.load(f)

                    # 只保留数值类型的 metrics，排除 split_id
                    filtered_results = {}
                    for key, value in split_results.items():
                        if key != 'split_id' and isinstance(value, (int, float)):
                            filtered_results[key] = value

                    results[split_id] = filtered_results
                    logger.info(f"Loaded results for split {split_id}")
                except Exception as e:
                    logger.warning(f"Failed to load results for split {split_id}: {e}")
            else:
                logger.warning(f"Results file not found for split {split_id}: {results_file}")

        return results

    def _load_log(self) -> Dict[str, Any]:
        """加载训练日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load training log: {e}")
        return {"splits": {}, "created_at": datetime.now().isoformat()}

    def _save_log(self, log_data: Dict[str, Any]):
        """保存训练日志"""
        log_data["updated_at"] = datetime.now().isoformat()
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _update_split_log(self, split_id: int, status: str, message: str = ""):
        """更新单个 split 的日志"""
        log_data = self._load_log()
        if str(split_id) not in log_data["splits"]:
            log_data["splits"][str(split_id)] = {}
        log_data["splits"][str(split_id)]["status"] = status
        log_data["splits"][str(split_id)]["updated_at"] = datetime.now().isoformat()
        if message:
            log_data["splits"][str(split_id)]["message"] = message
        self._save_log(log_data)

    def is_split_completed(self, split_id: int) -> bool:
        """检查 split 是否已完成"""
        split_output_dir = self.training_dir / f"split_{split_id}"
        results_file = split_output_dir / "training_results.json"
        return results_file.exists()

    def get_training_status(self) -> Dict[int, str]:
        """
        Get training status for all splits

        Returns:
            Dictionary mapping split_id to status ('pending', 'running', 'completed', 'failed')
        """
        status = {}

        for split_dir in sorted(self.training_dir.glob("split_*")):
            split_id = int(split_dir.name.split('_')[1])
            results_file = split_dir / "training_results.json"

            if results_file.exists():
                status[split_id] = 'completed'
            elif (split_dir / "training.log").exists():
                status[split_id] = 'running'
            else:
                status[split_id] = 'pending'

        return status
