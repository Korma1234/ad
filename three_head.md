1.方法称述
对于长期时序预测任务，将其拆解为二个任务，近期时间点的预报和远期时间点的预报。对于这两部分的预报有一个猜想，对于短期预报来说，时序特征中的趋势特征、季节特征和残差特征都很重要，而对于远期预报来说，趋势特征和季节特征的占比会更高。对于该猜想，首先要有一个主干模型去正常的提取时序特征，然后设计任务特征化的特征调整器，得到针对这两组任务的特征，然后各自去预测，最后再基于DS证据理论从可解释性角度去集成两部分的预测结果，以获得综合更好的预测结果。
2.相关代码：
（1）模型的代码integrated_three_head_model.py：
#!/usr/bin/env python3
"""
集成三头Time-MoE模型
基于ThreeHeadTimeMoeForPrediction，添加多种融合方法支持
- Early头: 只计算前32步的损失
- Late头: 只计算后32步的损失  
- All头: 计算全部96步的损失
- 融合模块: 支持MLP、注意力、DS等多种融合方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union, List, Dict
from transformers import PreTrainedModel
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.models.configuration_time_moe import TimeMoeConfig
from custom_model import CustomBenchmarkDataset
from decompose_simple import OptimizedTimeSeriesDecomposition
from ds_fusion import (
    DSFusionConfig, EvidenceNet, SimpleDSStyleFusion, SoftDSFusion,
    ReliabilityEMA, LearnedReliability, MLPWeightFusion, AttentionFusion, LearnableWeightFusion,
    rank_consistency_loss, ignorance_calibration_loss, diversity_regularizer, compute_batch_error
)


class IntegratedThreeHeadTimeMoeForPrediction(PreTrainedModel):
    """
    集成三头Time-MoE预测模型
    在原有三头模型基础上添加多种融合方法支持
    """
    
    def __init__(
        self, 
        config: TimeMoeConfig, 
        horizon_length: int = 96, 
        max_length: int = 512,
        fusion_method: str = "mlp",  # "mlp", "attention", "learnable", "ds_simple", "ds_soft"
        fusion_config: Optional[DSFusionConfig] = None
    ):
        super().__init__(config)
        
        self.config = config
        self.horizon_length = horizon_length
        self.max_length = max_length
        self.input_size = config.input_size
        self.fusion_method = fusion_method
        
        # 加载预训练的Time-MoE主干网络
        from time_moe.models.modeling_time_moe import TimeMoeModel
        self.model = TimeMoeModel(config)
        
        # 冻结主干网络参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 创建优化的时序分解模块
        self.decomposition = OptimizedTimeSeriesDecomposition(
            feature_dim=config.hidden_size,
            seq_len=max_length,
            num_heads=8,
            dropout=0.1,
            seasonal_periods=[24, 168]  # 日周期和周周期
        )
        
        # 创建三个独立的预测头
        self.early_head = nn.Linear(config.hidden_size, horizon_length * self.input_size, bias=False)
        self.late_head = nn.Linear(config.hidden_size, horizon_length * self.input_size, bias=False)
        self.all_head = nn.Linear(config.hidden_size, horizon_length * self.input_size, bias=False)
        
        # 初始化融合模块
        self._init_fusion_modules(fusion_config)
        
        # 训练步数计数器
        self.training_step = 0
        
        # 损失函数
        self.loss_function = torch.nn.MSELoss(reduction='none')
        
        # 损失权重参数 - 调整以平衡三个头的贡献
        self.alpha = 2.0  # Early头权重 (提高，因为表现稳定)
        self.beta = 1.5   # Late头权重 (大幅提高，因为表现最差)
        self.gamma = 0.8  # All头权重 (降低，因为过度主导)
        
        # 初始化权重
        self.post_init()
        
        print(f"🎯 集成三头Time-MoE模型初始化完成")
        print(f"   预测长度: {horizon_length}")
        print(f"   序列长度: {max_length}")
        print(f"   融合方法: {fusion_method}")
        print(f"   特征分解: 优化版本 (OptimizedTimeSeriesDecomposition)")
        print(f"   输入维度: {self.input_size}")
        print(f"   隐藏维度: {config.hidden_size}")
        print(f"   Early头: 只计算前{horizon_length//3}步损失 (权重α={self.alpha})")
        print(f"   Late头: 只计算后{horizon_length//3}步损失 (权重β={self.beta})")
        print(f"   All头: 计算全部{horizon_length}步损失 (权重γ={self.gamma})")
        print(f"   主干网络: 已冻结")
    
    def load_pretrained_weights(self, model_path: str):
        """
        加载预训练权重到主干网络
        参照ThreeHeadTimeMoeForPrediction的实现
        """
        print(f"🔄 加载预训练权重: {model_path}")
        
        try:
            # 尝试加载原始TimeMoeForPrediction模型
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            pretrained_model = TimeMoeForPrediction.from_pretrained(model_path)
            
            # 复制主干网络权重
            self.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
            print("✅ 预训练权重加载完成")
            
        except Exception as e:
            print(f"⚠️ 预训练权重加载失败: {e}")
            print("将从头开始训练")
    
    def _init_fusion_modules(self, fusion_config: Optional[DSFusionConfig]):
        """初始化融合模块"""
        if fusion_config is None:
            fusion_config = DSFusionConfig(
                num_experts=3,
                hidden_size=self.config.hidden_size,
                evidence_hidden=128,
                d_out=self.input_size,
                omega_mode="variance",
                use_reliability=True,
                reliability_mode="ema"
            )
        
        self.fusion_config = fusion_config
        
        if self.fusion_method == "mlp":
            # 简单MLP权重融合
            self.fusion_module = MLPWeightFusion(
                hidden_size=self.config.hidden_size,
                num_experts=3
            )
            
        elif self.fusion_method == "attention":
            # 基于注意力的融合
            self.fusion_module = AttentionFusion(
                hidden_size=self.config.hidden_size,
                num_experts=3,
                num_heads=4
            )
            
        elif self.fusion_method == "learnable":
            # 可学习权重融合
            self.fusion_module = LearnableWeightFusion(
                hidden_size=self.config.hidden_size,
                num_experts=3
            )
            
        elif self.fusion_method in ["ds_simple", "ds_soft"]:
            # DS证据融合
            self.evidence_net = EvidenceNet(
                hidden_size=self.config.hidden_size,
                num_experts=3,
                evidence_hidden=fusion_config.evidence_hidden,
                omega_mode=fusion_config.omega_mode
            )
            
            if self.fusion_method == "ds_simple":
                self.fusion_module = SimpleDSStyleFusion(
                    num_experts=3,
                    omega_mode=fusion_config.omega_mode,
                    omega_scale=fusion_config.omega_scale
                )
            else:  # ds_soft
                self.fusion_module = SoftDSFusion(
                    num_experts=3,
                    gamma=fusion_config.soft_ds_gamma,
                    eps=fusion_config.soft_ds_eps
                )
            
            # 可靠性估计
            if fusion_config.use_reliability:
                if fusion_config.reliability_mode == "ema":
                    self.reliability_ema = ReliabilityEMA(
                        num_experts=3,
                        alpha=fusion_config.ema_alpha,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                elif fusion_config.reliability_mode == "learned":
                    self.learned_reliability = LearnedReliability(
                        feature_dim=self.config.hidden_size,
                        num_experts=3
                    )
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        inputs: torch.FloatTensor = None,  # 添加inputs参数用于我们的数据集格式
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # 接受额外参数
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 处理参数映射：我们的数据集返回'inputs'但TimeMoeModel期望'input_ids'
        if input_ids is None and inputs is not None:
            input_ids = inputs
        
        # 从主干网络获取隐藏状态
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 特征分解：分别得到短期和长期特化特征
        short_decomposed_features = self.decomposition(hidden_states, horizon="short")  # [batch_size, seq_len, hidden_size]
        long_decomposed_features = self.decomposition(hidden_states, horizon="long")    # [batch_size, seq_len, hidden_size]
        
        # 取分解后特征的最后一个时间步
        original_last_hidden = hidden_states[:, -1, :]           # [batch_size, hidden_size]
        # short_last_hidden = short_decomposed_features # [batch_size, hidden_size]
        # long_last_hidden = long_decomposed_features   # [batch_size, hidden_size]
        short_last_hidden = short_decomposed_features + original_last_hidden # [batch_size, hidden_size]
        long_last_hidden = long_decomposed_features + original_last_hidden   # [batch_size, hidden_size]
        
        # 三个预测头的前向传播
        early_logits = self.early_head(short_last_hidden)  # [batch_size, horizon_length * input_size]
        late_logits = self.late_head(long_last_hidden)     # [batch_size, horizon_length * input_size]
        all_logits = self.all_head(original_last_hidden)   # [batch_size, horizon_length * input_size]
        
        # 重塑为预测形状
        batch_size = early_logits.size(0)
        early_predictions = early_logits.view(batch_size, self.horizon_length, self.input_size)
        late_predictions = late_logits.view(batch_size, self.horizon_length, self.input_size)
        all_predictions = all_logits.view(batch_size, self.horizon_length, self.input_size)
        
        # 堆叠专家预测
        expert_predictions = torch.stack([early_predictions, late_predictions, all_predictions], dim=2)  # [B, T, E, D]
        
        # 应用融合方法
        fused_predictions, fusion_info = self._apply_fusion(
            expert_predictions, original_last_hidden, hidden_states
        )
        
        # 计算损失
        total_loss = None
        if labels is not None:
            total_loss = self._compute_integrated_losses(
                early_predictions, late_predictions, all_predictions, 
                fused_predictions, labels, fusion_info
            )
        
        if not return_dict:
            # 始终返回(loss, predictions)以保持一致性
            return (total_loss, fused_predictions)
            
        return {
            'loss': total_loss,
            'logits': fused_predictions,  # 使用融合后的预测作为主要输出
            'early_predictions': early_predictions,
            'late_predictions': late_predictions,
            'all_predictions': all_predictions,
            'fused_predictions': fused_predictions,
            'fusion_info': fusion_info,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions if output_attentions else None,
        }
    
    def _apply_fusion(
        self, 
        expert_predictions: torch.Tensor, 
        last_hidden: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> tuple:
        """
        应用指定的融合方法
        
        Args:
            expert_predictions: [B, T, E, D] 专家预测
            last_hidden: [B, H] 最后时间步的隐藏状态
            hidden_states: [B, T, H] 完整隐藏状态
            
        Returns:
            fused_predictions: [B, T, D] 融合后的预测
            fusion_info: dict 融合信息
        """
        if self.fusion_method in ["mlp", "learnable"]:
            # 简单融合方法
            fusion_output = self.fusion_module(expert_predictions, last_hidden)
            fused_predictions = fusion_output["fused"]
            fusion_info = {
                "method": self.fusion_method,
                "weights": fusion_output["weights"],
                "uncertainty": fusion_output["uncertainty"]
            }
            
        elif self.fusion_method == "attention":
            # 注意力融合
            fusion_output = self.fusion_module(expert_predictions, hidden_states)
            fused_predictions = fusion_output["fused"]
            fusion_info = {
                "method": self.fusion_method,
                "weights": fusion_output["weights"],
                "attention_weights": fusion_output["attention_weights"],
                "uncertainty": fusion_output["uncertainty"]
            }
            
        elif self.fusion_method in ["ds_simple", "ds_soft"]:
            # DS证据融合
            evidence_logits, omega_logits = self.evidence_net(hidden_states, expert_predictions)
            
            # 获取可靠性
            reliability = None
            if self.fusion_config.use_reliability:
                if self.fusion_config.reliability_mode == "ema" and hasattr(self, 'reliability_ema'):
                    reliability = self.reliability_ema.get().view(1, 1, -1)  # [1, 1, 3]
                    reliability = reliability.expand(expert_predictions.size(0), expert_predictions.size(1), -1)  # [B, T, 3]
                elif self.fusion_config.reliability_mode == "learned" and hasattr(self, 'learned_reliability'):
                    reliability = self.learned_reliability(hidden_states)  # [B, T, 3]
            
            # 应用DS融合
            if self.fusion_method == "ds_simple":
                fusion_output = self.fusion_module(expert_predictions, evidence_logits, omega_logits)
            else:  # ds_soft
                fusion_output = self.fusion_module(expert_predictions, evidence_logits, reliability)
            
            fused_predictions = fusion_output["fused"]
            fusion_info = {
                "method": self.fusion_method,
                "evidence_logits": evidence_logits,
                "omega_logits": omega_logits,
                "betp": fusion_output["betp"],  # Pignistic概率
                "uncertainty": fusion_output["uncert"],
                "reliability": reliability
            }
            
            if self.fusion_method == "ds_soft":
                fusion_info["conflict"] = fusion_output.get("conflict", None)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_predictions, fusion_info
    
    def _compute_integrated_losses(
        self,
        early_predictions: torch.Tensor,
        late_predictions: torch.Tensor,
        all_predictions: torch.Tensor,
        fused_predictions: torch.Tensor,
        labels: torch.Tensor,
        fusion_info: Dict
    ) -> torch.Tensor:
        """计算集成损失"""
        
        # 1. 主预测损失（融合结果）
        main_loss = F.mse_loss(fused_predictions, labels)
        
        # 2. 个体专家损失（可选，用于稳定训练）
        early_length = self.horizon_length // 3
        late_start = 2 * self.horizon_length // 3
        
        early_loss = torch.mean(self.loss_function(early_predictions[:, :early_length, :], labels[:, :early_length, :]))
        late_loss = torch.mean(self.loss_function(late_predictions[:, late_start:, :], labels[:, late_start:, :]))
        all_loss = torch.mean(self.loss_function(all_predictions, labels))
        
        individual_loss = self.alpha * early_loss + self.beta * late_loss + self.gamma * all_loss
        
        # 3. 融合特定损失
        fusion_loss = torch.tensor(0.0, device=main_loss.device)
        
        if self.fusion_method in ["ds_simple", "ds_soft"]:
            # DS融合的额外损失
            expert_predictions = torch.stack([early_predictions, late_predictions, all_predictions], dim=2)
            
            # 排序一致性损失
            if "betp" in fusion_info:
                rank_loss = rank_consistency_loss(
                    fusion_info["betp"], expert_predictions, labels
                ) * self.fusion_config.lambda_rank
                fusion_loss += rank_loss
            
            # 无知校准损失
            if "uncertainty" in fusion_info and "ignorance" in fusion_info["uncertainty"]:
                calib_loss = ignorance_calibration_loss(
                    fusion_info["uncertainty"]["ignorance"], expert_predictions, labels
                ) * self.fusion_config.lambda_calib
                fusion_loss += calib_loss
            
            # 多样性正则化
            if "betp" in fusion_info:
                div_loss = diversity_regularizer(fusion_info["betp"]) * self.fusion_config.lambda_div
                fusion_loss += div_loss
        
        # 总损失
        total_loss = main_loss + 0.1 * individual_loss + fusion_loss
        
        return total_loss
    
    def update_reliability(self, expert_predictions: torch.Tensor, labels: torch.Tensor):
        """更新EMA可靠性估计（仅用于DS融合）"""
        if (self.fusion_method in ["ds_simple", "ds_soft"] and 
            hasattr(self, 'reliability_ema')):
            
            # 计算批次误差
            batch_err_mean = compute_batch_error(expert_predictions, labels)
            self.reliability_ema.update(batch_err_mean.detach())
    
    def get_fusion_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """获取当前融合权重（用于分析）"""
        with torch.no_grad():
            last_hidden = hidden_states[:, -1, :]
            
            if self.fusion_method == "mlp":
                return self.fusion_module.fusion_net(last_hidden)
            elif self.fusion_method == "learnable":
                static_weights = F.softmax(self.fusion_module.learnable_weights, dim=-1)
                dynamic_weights = self.fusion_module.weight_net(last_hidden)
                return 0.7 * static_weights.unsqueeze(0) + 0.3 * dynamic_weights
            else:
                # 对于其他方法，返回平均权重
                return torch.ones(hidden_states.size(0), 3, device=hidden_states.device) / 3


class IntegratedThreeHeadTimeMoeRunner:
    """
    集成三头Time-MoE模型训练器
    基于ThreeHeadTimeMoeRunner，添加融合方法支持
    """
    
    def __init__(
        self,
        data_path: str,
        model_path: str,
        output_path: str,
        fusion_method: str = "mlp",
        horizon_length: int = 96,
        max_length: int = 512,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-5,
        micro_batch_size: int = 4,
        global_batch_size: int = 32,
        logging_steps: int = 10,
        evaluation_strategy: str = "no",
        eval_steps: int = 100,
        dataloader_num_workers: int = 16,
        precision: str = "fp32",
        seed: int = 9989,
        freeze_backbone: bool = True,
        mix_test_samples: bool = False,
        test_mix_ratio: float = 0.1,
        test_mix_seed: int = None,
        fusion_config: Optional[DSFusionConfig] = None,
        **kwargs
    ):
        # 导入原始训练器
        from three_head_model import ThreeHeadTimeMoeRunner
        
        # 初始化原始训练器
        self.base_runner = ThreeHeadTimeMoeRunner(
            data_path=data_path,
            model_path=model_path,
            output_path=output_path,
            horizon_length=horizon_length,
            max_length=max_length,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            dataloader_num_workers=dataloader_num_workers,
            precision=precision,
            seed=seed,
            freeze_backbone=freeze_backbone,
            mix_test_samples=mix_test_samples,
            test_mix_ratio=test_mix_ratio,
            test_mix_seed=test_mix_seed,
            **kwargs
        )
        
        # 添加融合相关参数
        self.fusion_method = fusion_method
        self.fusion_config = fusion_config or DSFusionConfig()
        
        print(f"🔧 集成模型训练器初始化")
        print(f"   融合方法: {fusion_method}")
        print(f"   融合配置: {self.fusion_config}")
    
    def create_model(self):
        """创建集成模型"""
        # 加载配置
        config = TimeMoeConfig.from_pretrained(self.base_runner.model_path)
        
        # 创建集成模型
        model = IntegratedThreeHeadTimeMoeForPrediction(
            config=config,
            horizon_length=self.base_runner.horizon_length,
            max_length=self.base_runner.max_length,
            fusion_method=self.fusion_method,
            fusion_config=self.fusion_config
        )
        
        # 加载预训练权重 - 这是关键！
        model.load_pretrained_weights(self.base_runner.model_path)
        
        return model
    
    def train_model(self):
        """训练集成模型"""
        print(f"🚀 开始训练集成模型 (融合方法: {self.fusion_method})")
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据集
        train_dataset = self.base_runner.get_train_dataset()
        eval_dataset = self.base_runner.get_eval_dataset()
        
        # 创建训练器
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=self.base_runner.output_path,
            num_train_epochs=self.base_runner.num_epochs,
            per_device_train_batch_size=self.base_runner.micro_batch_size,
            per_device_eval_batch_size=self.base_runner.micro_batch_size,
            gradient_accumulation_steps=self.base_runner.global_batch_size // (self.base_runner.micro_batch_size * 8),  # 8 GPUs
            learning_rate=self.base_runner.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-08,
            max_grad_norm=1.0,
            logging_steps=self.base_runner.logging_steps,
            evaluation_strategy=self.base_runner.evaluation_strategy,
            eval_steps=self.base_runner.eval_steps,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=False,  # 与原始代码保持一致
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=self.base_runner.dataloader_num_workers,
            fp16=(self.base_runner.precision == "fp16"),
            bf16=(self.base_runner.precision == "bf16"),
            seed=self.base_runner.seed,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # 创建自定义训练器，使用正确的学习率调度器实现
        class MinLRTrainer(Trainer):
            def __init__(self, min_learning_rate, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.min_learning_rate = min_learning_rate
            
            def create_scheduler(self, num_training_steps, optimizer=None):
                optimizer = self.optimizer if optimizer is None else optimizer
                min_lr_ratio = self.min_learning_rate / self.args.learning_rate
                
                if self.lr_scheduler is None:
                    if self.args.lr_scheduler_type == 'cosine':
                        self.lr_scheduler = self._get_cosine_schedule_with_warmup_min_lr(
                            optimizer=optimizer,
                            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                            num_training_steps=num_training_steps,
                            min_lr_ratio=min_lr_ratio,
                        )
                    else:
                        from transformers import get_scheduler
                        self.lr_scheduler = get_scheduler(
                            self.args.lr_scheduler_type,
                            optimizer=optimizer,
                            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                            num_training_steps=num_training_steps,
                        )
                    self._created_lr_scheduler = True
                return self.lr_scheduler
            
            def _get_cosine_schedule_with_warmup_min_lr(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr_ratio=0, last_epoch=-1):
                """
                创建带有最小学习率的余弦退火调度器
                """
                import math
                from torch.optim.lr_scheduler import LambdaLR
                
                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
                
                return LambdaLR(optimizer, lr_lambda, last_epoch)
            
            def log(self, logs):
                # 记录当前学习率
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    current_lr = self.args.learning_rate
                
                logs['learning_rate'] = current_lr
                
                # 调用父类的log方法
                super().log(logs)
                
                # 额外记录学习率到文件
                if self.state.global_step % 10 == 0:
                    with open('integrated_3head_training_loss_log.txt', 'a', encoding='utf-8') as f:
                        f.write(f"📈 Step {self.state.global_step}: 学习率 = {current_lr:.8f}\n")
        
        trainer = MinLRTrainer(
            min_learning_rate=self.base_runner.min_learning_rate,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        print(f"💾 模型已保存到: {self.base_runner.output_path}")
        
        return trainer, model
    
    def get_train_dataset(self):
        """获取训练数据集"""
        return self.base_runner.get_train_dataset()
    
    def get_eval_dataset(self):
        """获取评估数据集"""
        return self.base_runner.get_eval_dataset()


def evaluate_integrated_model(model, eval_dataset, device, batch_size=32, max_samples=0):
    """
    评估集成三头模型：
    - 分别评估 Early/Late/All 三个头的分段与整体性能
    - 评估融合预测的分段与整体性能
    输出包含四路（early/late/all/fused）的 overall/early/late 的 MSE/MAE 指标
    """
    import json
    model.eval()

    early_length = model.horizon_length // 3
    late_start = 2 * model.horizon_length // 3

    # 限制评估样本数
    total_samples = len(eval_dataset)
    if max_samples is not None and max_samples > 0:
        total_samples = min(total_samples, max_samples)

    # 收集预测
    all_early, all_late, all_all, all_fused, all_labels = [], [], [], [], []

    with torch.no_grad():
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_inputs = []
            batch_labels = []

            for i in range(batch_start, batch_end):
                sample = eval_dataset[i]
                batch_inputs.append(torch.from_numpy(sample['inputs']))
                batch_labels.append(torch.from_numpy(sample['labels']))

            inputs = torch.stack(batch_inputs).to(device)      # [B, Tin, Din]
            labels = torch.stack(batch_labels).to(device)      # [B, Tout, Dout]

            outputs = model(inputs=inputs)  # forward 返回 dict

            all_early.append(outputs['early_predictions'].detach().cpu())
            all_late.append(outputs['late_predictions'].detach().cpu())
            all_all.append(outputs['all_predictions'].detach().cpu())
            # 兼容键名（有的代码用 logits，有的用 fused_predictions）
            fused = outputs.get('fused_predictions', None)
            if fused is None:
                fused = outputs.get('logits')
            all_fused.append(fused.detach().cpu())
            all_labels.append(labels.detach().cpu())

    early_preds = torch.cat(all_early, dim=0)
    late_preds = torch.cat(all_late, dim=0)
    all_preds = torch.cat(all_all, dim=0)
    fused_preds = torch.cat(all_fused, dim=0)
    labels = torch.cat(all_labels, dim=0)

    def compute_metrics(preds, labs):
        metrics = {}
        metrics['overall_mse'] = torch.mean((preds - labs) ** 2).item()
        metrics['overall_mae'] = torch.mean(torch.abs(preds - labs)).item()
        metrics['early_mse'] = torch.mean((preds[:, :early_length, :] - labs[:, :early_length, :]) ** 2).item()
        metrics['early_mae'] = torch.mean(torch.abs(preds[:, :early_length, :] - labs[:, :early_length, :])).item()
        metrics['late_mse'] = torch.mean((preds[:, late_start:, :] - labs[:, late_start:, :]) ** 2).item()
        metrics['late_mae'] = torch.mean(torch.abs(preds[:, late_start:, :] - labs[:, late_start:, :])).item()
        return metrics

    results = {
        'early_head': compute_metrics(early_preds, labels),
        'late_head': compute_metrics(late_preds, labels),
        'all_head': compute_metrics(all_preds, labels),
        'fused': compute_metrics(fused_preds, labels),
        'fusion_method': getattr(model, 'fusion_method', None),
        'fusion_config': getattr(model, 'fusion_config', None).__dict__ if hasattr(model, 'fusion_config') else None,
    }

    # 保存
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
（2）模块代码ds_fusion.py和decompose_simple.py：
1)
#!/usr/bin/env python3
"""
DS证据融合模块
基于Dempster-Shafer证据理论的多专家融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class DSFusionConfig:
    """DS融合配置"""
    num_experts: int = 3
    hidden_size: int = 256
    evidence_hidden: int = 128
    d_out: int = 1
    omega_mode: str = "variance"     # "variance" | "learned" | "constant"
    omega_scale: float = 1.0
    use_reliability: bool = True
    reliability_mode: str = "ema"    # "ema" | "learned" | "none"
    ema_alpha: float = 0.9
    soft_ds_gamma: float = 0.8
    soft_ds_eps: float = 1e-6
    
    # Loss 权重
    lambda_rank: float = 0.2
    lambda_calib: float = 0.1
    lambda_div: float = 0.0


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class EvidenceNet(nn.Module):
    """证据网络：由隐藏状态输出专家证据logits"""
    def __init__(self, hidden_size: int, num_experts: int, evidence_hidden: int = 128, omega_mode: str = "variance"):
        super().__init__()
        self.num_experts = num_experts
        self.omega_mode = omega_mode
        
        # 证据生成网络
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, evidence_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(evidence_hidden, num_experts)
        )
        
        # Omega证据生成（可选）
        if omega_mode == "learned":
            self.omega_head = nn.Sequential(
                nn.Linear(hidden_size, evidence_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(evidence_hidden, 1)
            )
        else:
            self.omega_head = None
        
        self.pos_enc = PositionalEncoding(hidden_size)

    def forward(self, Z: torch.Tensor, expert_preds: Optional[torch.Tensor] = None):
        """
        Z: [B,seq_len,H] 隐藏状态
        expert_preds: [B,T,E,D] 专家预测（可选，用于variance模式）
        """
        if expert_preds is not None:
            B, T, E, D = expert_preds.shape
            # 使用Z的最后一个时间步作为全局特征
            last_hidden = Z[:, -1, :]  # [B, H]
            # 广播到所有预测时间步
            Z_expanded = last_hidden.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]
        else:
            Z_expanded = Z
        
        Zc = self.pos_enc(Z_expanded)
        evidence_logits = self.proj(Zc)  # [B,T,E]
        
        omega_logits = None
        if self.omega_mode == "learned":
            omega_logits = self.omega_head(Zc)  # [B,T,1]
        
        return evidence_logits, omega_logits


class ReliabilityEMA:
    """EMA可靠性估计"""
    def __init__(self, num_experts: int, alpha: float = 0.9, device: str = "cpu"):
        self.alpha = alpha
        self.num_experts = num_experts
        self.device = device
        self.register()

    def register(self):
        self.running = torch.ones(self.num_experts, device=self.device)

    def update(self, batch_err_mean: torch.Tensor):
        # batch_err_mean: [E]
        self.running = self.alpha * self.running + (1 - self.alpha) * batch_err_mean

    def get(self) -> torch.Tensor:
        inv = 1.0 / (self.running + 1e-6)
        r = inv / (inv.sum() + 1e-9)  # [E]
        return r


class LearnedReliability(nn.Module):
    """可学习可靠性估计"""
    def __init__(self, feature_dim: int, num_experts: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_experts),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B,T,F]
        return: [B,T,E] in (0,1)
        """
        return self.net(feat)


class SimpleDSStyleFusion(nn.Module):
    """简单DS风格融合（不执行迭代组合）"""
    def __init__(self, num_experts: int, omega_mode: str = "variance", omega_scale: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.omega_mode = omega_mode
        self.omega_scale = omega_scale

    def forward(self, expert_preds: torch.Tensor, evidence_logits: torch.Tensor, omega_logits: Optional[torch.Tensor] = None):
        """
        expert_preds:    [B,T,E,D]
        evidence_logits: [B,T,E]
        omega_logits:    [B,T,1]  (learned 模式)
        """
        B, T, E, D = expert_preds.shape
        evidence = F.softplus(evidence_logits)  # 保证非负

        # 生成 Omega 证据
        if self.omega_mode == "learned":
            assert omega_logits is not None, "omega_mode=learned 需传入 omega_logits"
            e_omega = F.softplus(omega_logits)
        elif self.omega_mode == "variance":
            # 使用专家预测的方差作为无知证据
            mean_each = expert_preds.mean(dim=-1)          # [B,T,E]
            var = torch.var(mean_each, dim=2, keepdim=True)  # [B,T,1]
            e_omega = self.omega_scale * (var + 1e-6)
        elif self.omega_mode == "constant":
            e_omega = torch.ones(B, T, 1, device=expert_preds.device)
        else:
            raise ValueError("Unknown omega_mode")

        # 归一化
        denom = evidence.sum(dim=-1, keepdim=True) + e_omega  # [B,T,1]
        m_experts = evidence / (denom + 1e-9)                 # [B,T,E]
        m_omega = e_omega / (denom + 1e-9)                   # [B,T,1]

        # Pignistic 概率
        betp = m_experts + m_omega / self.num_experts        # [B,T,E]
        fused = torch.einsum("bte,bted->btd", betp, expert_preds)

        out = {
            "fused": fused,          # [B,T,D]
            "betp": betp,            # [B,T,E]
            "m_experts": m_experts,  # [B,T,E]
            "m_omega": m_omega,      # [B,T,1]
            "uncert": {
                "ignorance": m_omega,
                "entropy": -(betp * (betp + 1e-9).log()).sum(-1, keepdim=True),
                "divergence": torch.var(mean_each, dim=2, keepdim=True)
            }
        }
        return out


class SoftDSFusion(nn.Module):
    """Soft-DS迭代组合融合（冲突感知）"""
    def __init__(self, num_experts: int, gamma: float = 0.8, eps: float = 1e-6):
        super().__init__()
        self.num_experts = num_experts
        self.gamma = gamma
        self.eps = eps

    def forward(self, expert_preds: torch.Tensor, evidence_logits: torch.Tensor, reliability: Optional[torch.Tensor] = None):
        """
        expert_preds:    [B,T,E,D]
        evidence_logits: [B,T,E]
        reliability:     [B,T,E] (前置折扣，可 None)
        """
        B, T, E, D = expert_preds.shape
        evidence = F.softplus(evidence_logits)
        if reliability is not None:
            evidence = evidence * torch.clamp(reliability, 0.0, 1.0)

        p = evidence / (evidence.sum(dim=-1, keepdim=True) + 1e-9)  # [B,T,E]

        # 构造单源
        sources: List[Dict[int, torch.Tensor]] = []
        for i in range(E):
            src = {}
            for j in range(E):
                if j == i:
                    src[j] = p[..., i]  # [B,T]
                else:
                    src[j] = torch.zeros_like(p[..., i])  # 使用p[..., i]而不是p[..., 0]
            src['Omega'] = 1 - p[..., i]
            sources.append(src)

        def combine_pair(m1, m2):
            # m1,m2: dict {0:[B,T],1:[B,T],..., 'Omega':[B,T]}
            K = torch.zeros(B, T, device=p.device)
            for i in range(E):
                for j in range(E):
                    if i != j:
                        K += m1[i] * m2[j]
            Dn = (1 - K).clamp(min=0)
            D = torch.pow(Dn, self.gamma) + self.eps
            m_new = {}
            for k in range(E):
                num = m1[k]*m2[k] + m1[k]*m2['Omega'] + m1['Omega']*m2[k]
                m_new[k] = num / D
            m_new['Omega'] = (m1['Omega'] * m2['Omega']) / D
            return m_new, K

        # 迭代组合
        current = sources[0]
        K_list = []
        for i in range(1, E):
            current, K = combine_pair(current, sources[i])
            K_list.append(K.unsqueeze(-1))
        K_all = torch.cat(K_list, dim=-1) if K_list else torch.zeros(B, T, 1, device=p.device)

        # Pignistic
        betp_list = [current[i] + current['Omega'] / E for i in range(E)]
        betp = torch.stack(betp_list, dim=-1)  # [B,T,E]
        fused = torch.einsum("bte,bted->btd", betp, expert_preds)

        mean_each = expert_preds.mean(dim=-1)  # [B,T,E]
        out = {
            "fused": fused,
            "betp": betp,
            "m_final": current,
            "conflict": K_all,  # 每次组合的 K
            "uncert": {
                "ignorance": current['Omega'].unsqueeze(-1),
                "entropy": -(betp * (betp + 1e-9).log()).sum(-1, keepdim=True),
                "divergence": torch.var(mean_each, dim=2, keepdim=True)
            }
        }
        return out


class MLPWeightFusion(nn.Module):
    """简单MLP权重融合"""
    def __init__(self, hidden_size: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_preds: torch.Tensor, last_hidden: torch.Tensor):
        """
        expert_preds: [B,T,E,D]
        last_hidden: [B,H]
        """
        weights = self.fusion_net(last_hidden)  # [B, E]
        weights = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, E, 1]
        fused = torch.sum(expert_preds * weights, dim=2)  # [B, T, D]
        
        return {
            "fused": fused,
            "weights": weights.squeeze(1).squeeze(-1),  # [B, E]
            "uncertainty": None
        }


class AttentionFusion(nn.Module):
    """基于注意力的融合 - 在专家维度上做自注意力，保持D维度不变"""
    def __init__(self, hidden_size: int, num_experts: int = 3, num_heads: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = 96 # pred len
        
        # 使用自注意力机制在专家维度上进行融合
        # 注意：这里需要将T维度作为embed_dim，因为我们要在E维度上做注意力
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=self.hidden_size,  # 这里应该是T维度，但MultiheadAttention需要固定维度
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合权重计算 - 输入维度应该是D，不是hidden_size
        self.weight_net = nn.Sequential(
            nn.Linear(1, 16),  # 输入是D=1，输出16维
            nn.ReLU(),
            nn.Linear(16, num_experts)  # 输出num_experts=3维
        )

    def forward(self, expert_preds: torch.Tensor, hidden_states: torch.Tensor):
        """
        expert_preds: [B,T,E,D]
        hidden_states: [B,T,H]
        """
        B, T, E, D = expert_preds.shape
        
        # 重塑为 [B*D, E, T] 进行专家维度的注意力
        # 将D维度与B维度合并，E作为序列长度，T作为特征维度
        expert_preds_reshaped = expert_preds.permute(0, 3, 2, 1)  # [B, D, E, T]
        expert_preds_flat = expert_preds_reshaped.contiguous().view(B*D, E, T)  # [B*D, E, T]
        
        # 自注意力：在专家维度上学习专家间的关系
        attn_output, attn_weights = self.attention_fusion(
            query=expert_preds_flat,  # [B*D, E, T]
            key=expert_preds_flat,    # [B*D, E, T] 
            value=expert_preds_flat   # [B*D, E, T]
        )
        
        # 重塑回原始形状
        attn_output = attn_output.view(B*D, E, T)  # [B*D, E, T]
        attn_output = attn_output.permute(0, 2, 1)  # [B*D, T, E]
        attn_output = attn_output.view(B, D, T, E)  # [B, D, T, E]
        attn_output = attn_output.permute(0, 2, 3, 1)  # [B, T, E, D]
        
        # 计算融合权重 - 基于注意力输出的平均
        attn_mean = attn_output.mean(dim=1)  # [B, E, D]
        attn_mean_flat = attn_mean.view(B*E, D)  # [B*E, D]
        fusion_weights_flat = self.weight_net(attn_mean_flat)  # [B*E, num_experts]
        fusion_weights = fusion_weights_flat.view(B, E, self.num_experts)  # [B, E, num_experts]
        fusion_weights = F.softmax(fusion_weights, dim=-1)
        
        # 应用权重 - 使用对角权重（每个专家对应自己的权重）
        expert_weights = torch.diagonal(fusion_weights, dim1=1, dim2=2)  # [B, E]
        weights = expert_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, E, 1]
        fused = torch.sum(attn_output * weights, dim=2)  # [B, T, D]
        
        # 计算不确定性（基于注意力权重的熵）
        uncertainty = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        uncertainty = uncertainty.mean(dim=1)  # [B*D]
        uncertainty = uncertainty.view(B, D).mean(dim=1)  # [B]
        
        return {
            "fused": fused,
            "weights": expert_weights,  # [B, E]
            "attention_weights": attn_weights,
            "uncertainty": uncertainty
        }


class LearnableWeightFusion(nn.Module):
    """可学习权重融合"""
    def __init__(self, hidden_size: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.learnable_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_preds: torch.Tensor, last_hidden: torch.Tensor):
        """
        expert_preds: [B,T,E,D]
        last_hidden: [B,H]
        """
        # 结合静态权重和动态权重
        static_weights = F.softmax(self.learnable_weights, dim=-1)  # [E]
        dynamic_weights = self.weight_net(last_hidden)  # [B, E]
        
        # 混合权重
        mixed_weights = 0.7 * static_weights.unsqueeze(0) + 0.3 * dynamic_weights  # [B, E]
        mixed_weights = F.softmax(mixed_weights, dim=-1)
        
        weights = mixed_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, E, 1]
        fused = torch.sum(expert_preds * weights, dim=2)  # [B, T, D]
        
        return {
            "fused": fused,
            "weights": mixed_weights,  # 添加weights键，使用mixed_weights
            "static_weights": static_weights,
            "dynamic_weights": dynamic_weights,
            "mixed_weights": mixed_weights,
            "uncertainty": None
        }


# 损失函数
def rank_consistency_loss(betp: torch.Tensor, expert_preds: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """排序一致性损失"""
    with torch.no_grad():
        err = ((expert_preds - target.unsqueeze(2)) ** 2).mean(dim=-1)  # [B,T,E]
    B, T, E = betp.shape
    terms = []
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            mask = (err[..., i] < err[..., j]).float()
            violation = F.relu(betp[..., j] - betp[..., i] + margin)
            terms.append(violation * mask)
    if not terms:
        return torch.tensor(0.0, device=betp.device)
    loss = torch.stack(terms, dim=0).mean()
    return loss


def ignorance_calibration_loss(ignorance: torch.Tensor, expert_preds: torch.Tensor, target: torch.Tensor, detach_pred: bool = True) -> torch.Tensor:
    """无知校准损失"""
    err = ((expert_preds.mean(dim=2) - target) ** 2).mean(dim=-1, keepdim=True)  # [B,T,1]
    if detach_pred:
        err = err.detach()
    ig = ignorance
    ig_c = ig - ig.mean()
    err_c = err - err.mean()
    num = (ig_c * err_c).mean()
    denom = (ig_c.pow(2).mean().sqrt() * err_c.pow(2).mean().sqrt() + 1e-9)
    corr = num / denom
    return (1 - corr).clamp(min=0.0)


def diversity_regularizer(betp: torch.Tensor) -> torch.Tensor:
    """多样性正则化"""
    entropy = -(betp * (betp + 1e-9).log()).sum(dim=-1)  # [B,T]
    return -entropy.mean()


def compute_batch_error(expert_preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算批次误差"""
    err = (expert_preds - target.unsqueeze(2)) ** 2  # [B,T,E,D]
    mse = err.mean(dim=(0, 1, 3))
    return mse

2)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedTimeSeriesDecomposition(nn.Module):
    def __init__(self, feature_dim, seq_len, num_heads=4, dropout=0.1, seasonal_periods=[24, 168]):
        """
        针对最后时间步预测优化的时序分解模块
        
        Args:
            feature_dim: 特征维度
            seq_len: 序列长度
            num_heads: 注意力头数量
            dropout: Dropout比率
            seasonal_periods: 季节性周期列表
        """
        super(OptimizedTimeSeriesDecomposition, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.seasonal_periods = seasonal_periods
        
        # 1. 轻量级趋势提取器 - 只关注与最后时间步相关的计算
        self.trend_extractor = LastStepTrendExtractor(
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 2. 聚焦型季节性提取器 - 只提取关键周期位置
        self.seasonal_extractor = LastStepSeasonalExtractor(
            feature_dim=feature_dim,
            seq_len=seq_len,
            periods=seasonal_periods,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 3. 局部残差提取器 - 只关注最近的时间步
        self.residual_extractor = LastStepResidualExtractor(
            feature_dim=feature_dim,
            window_size=min(10, seq_len-1),  # 最多看最近10个时间步
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 4. 简化的组件融合网络
        self.fusion_layer = OptimizedComponentFusion(feature_dim)
        
    def forward(self, x, horizon="short"):
        """
        前向传播
        
        Args:
            x: 基础模型输出的特征 [batch_size, seq_len, feature_dim]
            horizon: 预测长度类型，"short"或"long"
            
        Returns:
            final_features: 分解后的最后时间步特征 [batch_size, feature_dim]
        """
        # 提取最后时间步的三个组件
        trend_features = self.trend_extractor(x)
        seasonal_features = self.seasonal_extractor(x)
        residual_features = self.residual_extractor(x)
        
        # 融合组件
        weights = torch.ones(3, device=x.device)
        if horizon == "long":
            # 长期预测更关注趋势和季节性
            weights = torch.tensor([0.5, 0.4, 0.1], device=x.device)
        else:
            # 短期预测更平衡
            weights = torch.tensor([0.3, 0.3, 0.4], device=x.device)
            
        final_features = self.fusion_layer(
            trend_features, seasonal_features, residual_features, weights, "adaptive"
        )
        
        return final_features


class LastStepTrendExtractor(nn.Module):
    """优化的趋势提取器，只计算最后时间步所需的趋势特征"""
    
    def __init__(self, feature_dim, seq_len, num_heads=4, dropout=0.1):
        super(LastStepTrendExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        
        # 多头注意力 - 只计算最后一个查询向量
        self.attn = EfficientCausalAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 后处理网络
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        仅计算最后时间步的趋势特征
        
        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            trend_features: 最后时间步的趋势特征 [batch_size, feature_dim]
        """
        # 使用原始数据（不添加位置编码）
        x_pos = x
        
        # 提取最后时间步的查询向量
        last_query = x_pos[:, -1:, :]  # [batch_size, 1, feature_dim]
        
        # 应用趋势偏好的注意力 - 只计算最后一个查询
        trend_attn_output = self.attn(
            query=last_query,
            key_value=x_pos,
            is_trend=True
        )  # [batch_size, 1, feature_dim]
        
        # 残差连接和层归一化
        trend = self.norm(last_query + trend_attn_output)  # [batch_size, 1, feature_dim]
        
        # 前馈网络
        trend_ffn_output = self.ffn(trend)
        
        # 最终趋势特征
        trend_features = self.final_norm(trend + trend_ffn_output).squeeze(1)  # [batch_size, feature_dim]
        
        return trend_features


class LastStepSeasonalExtractor(nn.Module):
    """优化的季节性提取器，只计算最后时间步所需的季节性特征"""
    
    def __init__(self, feature_dim, seq_len, periods, num_heads=4, dropout=0.1):
        super(LastStepSeasonalExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.periods = periods
        
        # 每个周期的注意力模块
        self.attns = nn.ModuleList([
            EfficientCausalAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in periods
        ])
        
        # 后处理网络
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        仅计算最后时间步的季节性特征
        
        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            seasonal_features: 最后时间步的季节性特征 [batch_size, feature_dim]
        """
        # 提取最后时间步的查询向量
        last_query = x[:, -1:, :]  # [batch_size, 1, feature_dim]
        
        # 计算每个周期的季节性组件
        seasonal_outputs = []
        for i, period in enumerate(self.periods):
            # 选择关键的周期性位置
            indices = []
            last_idx = self.seq_len - 1
            
            # 从最后时间步向前，选择所有匹配周期的位置
            for j in range(min(5, self.seq_len // period)):  # 最多看5个周期
                idx = last_idx - period * (j + 1)
                if idx >= 0:
                    indices.append(idx)
            
            # 如果没有足够的历史，使用所有可用数据
            if not indices:
                indices = list(range(self.seq_len - 1))
                
            # 构建键值序列，只包含相关周期位置
            indices = [self.seq_len - 1] + indices  # 包含最后时间步自身
            key_value = x[:, indices, :]  # [batch_size, len(indices), feature_dim]
            
            # 应用季节性偏好的注意力
            seasonal_output = self.attns[i](
                query=last_query,
                key_value=key_value,
                is_trend=False,
                period=period
            )  # [batch_size, 1, feature_dim]
            
            seasonal_outputs.append(seasonal_output)
        
        # 合并多个季节性组件
        if len(seasonal_outputs) > 1:
            seasonal = sum(seasonal_outputs) / len(seasonal_outputs)
        else:
            seasonal = seasonal_outputs[0]
            
        # 残差连接和层归一化
        seasonal = self.norm(last_query + seasonal)
        
        # 前馈网络
        seasonal_ffn_output = self.ffn(seasonal)
        
        # 最终季节性特征
        seasonal_features = self.final_norm(seasonal + seasonal_ffn_output).squeeze(1)
        
        return seasonal_features


class LastStepResidualExtractor(nn.Module):
    """优化的残差提取器，只关注最近几个时间步"""
    
    def __init__(self, feature_dim, window_size=10, num_heads=4, dropout=0.1):
        super(LastStepResidualExtractor, self).__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 后处理网络
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        仅计算最后时间步的残差特征
        
        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            residual_features: 最后时间步的残差特征 [batch_size, feature_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 只保留最近的window_size+1个时间步（包括最后一步）
        recent_window = max(0, seq_len - self.window_size - 1)
        x_recent = x[:, recent_window:, :]  # [batch_size, min(seq_len, window_size+1), feature_dim]
        
        # 使用窗口数据（不添加位置编码）
        x_pos = x_recent
        
        # 提取最后时间步的查询向量
        last_query = x_pos[:, -1:, :]  # [batch_size, 1, feature_dim]
        
        # 创建残差掩码 - 只关注短期依赖
        recent_len = x_recent.size(1)
        residual_mask = torch.ones(1, recent_len, device=x.device) * float('-inf')
        
        # 最近window_size个时间步可见
        for i in range(min(self.window_size, recent_len-1)):
            # 距离越近权重越高
            residual_mask[0, -(i+2)] = 1.0 - 0.8 * i / self.window_size
        
        # 自身可见
        residual_mask[0, -1] = 0.0
        
        # 应用残差偏好的注意力
        residual_attn_output, _ = self.attn(
            query=last_query,
            key=x_pos,
            value=x_pos,
            attn_mask=residual_mask
        )  # [batch_size, 1, feature_dim]
        
        # 残差连接和层归一化
        residual = self.norm(last_query + residual_attn_output)
        
        # 前馈网络
        residual_ffn_output = self.ffn(residual)
        
        # 最终残差特征
        residual_features = self.final_norm(residual + residual_ffn_output).squeeze(1)
        
        return residual_features


class EfficientCausalAttention(nn.Module):
    """高效的因果注意力，只计算最后一个查询向量所需的注意力"""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(EfficientCausalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, is_trend=False, period=None):
        """
        高效的注意力计算，只关注最后一个查询向量
        
        Args:
            query: 查询向量，通常是最后时间步 [batch_size, 1, embed_dim]
            key_value: 键值序列 [batch_size, seq_len, embed_dim]
            is_trend: 是否为趋势注意力
            period: 季节性周期
            
        Returns:
            attn_output: 注意力输出 [batch_size, 1, embed_dim]
        """
        batch_size, seq_len, _ = key_value.shape
        assert query.size(1) == 1, "查询应该只包含一个时间步"
        
        # 线性变换
        q = self.q_proj(query)  # [batch_size, 1, embed_dim]
        k = self.k_proj(key_value)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(key_value)  # [batch_size, seq_len, embed_dim]
        
        # 分离头
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, seq_len]
        
        # 应用特定掩码
        if is_trend:
            # 趋势偏好 - 远距离权重增强
            weights = torch.ones(1, 1, 1, seq_len, device=query.device)
            for j in range(seq_len):
                # 最后一个时间步到其他步的距离
                dist = seq_len - 1 - j
                # 距离越远权重越高
                weights[0, 0, 0, j] = 1.0 + 0.5 * dist / seq_len
            
            attn_weights = attn_weights * weights
            
        elif period is not None:
            # 季节性偏好 - 周期位置增强
            weights = torch.ones(1, 1, 1, seq_len, device=query.device) * 0.1  # 基础权重低
            
            # 周期位置给予高权重
            for j in range(seq_len):
                # 计算与最后时间步的距离
                dist = seq_len - 1 - j
                if dist > 0 and dist % period == 0:
                    weights[0, 0, 0, j] = 1.0  # 周期位置
            
            attn_weights = attn_weights * weights
        
        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class OptimizedComponentFusion(nn.Module):
    """优化的组件融合模块，专注于最后时间步"""
    
    def __init__(self, feature_dim):
        super(OptimizedComponentFusion, self).__init__()
        
        # 组件变换
        self.trend_transform = nn.Linear(feature_dim, feature_dim)
        self.seasonal_transform = nn.Linear(feature_dim, feature_dim)
        self.residual_transform = nn.Linear(feature_dim, feature_dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 动态权重调整网络（可选，仅在需要时使用）
        self.weight_adjust = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, trend, seasonal, residual, weights, horizon):
        """
        融合三个组件
        
        Args:
            trend: 趋势特征 [batch_size, feature_dim]
            seasonal: 季节性特征 [batch_size, feature_dim]
            residual: 残差特征 [batch_size, feature_dim]
            weights: 默认权重 [3]
            horizon: 预测类型
            
        Returns:
            fused: 融合后的特征 [batch_size, feature_dim]
        """
        batch_size, feature_dim = trend.shape
        
        # 动态权重调整（可选）
        if horizon == "adaptive":
            # 连接特征评估重要性
            combined = torch.cat([trend, seasonal, residual], dim=-1)
            dynamic_weights = self.weight_adjust(combined)  # [batch_size, 3]
            
            # 结合静态和动态权重
            batch_weights = torch.zeros(batch_size, 3, device=trend.device)
            for b in range(batch_size):
                batch_weights[b] = (weights + dynamic_weights[b]) / 2
        else:
            # 使用静态权重
            batch_weights = weights.expand(batch_size, 3)
        
        # 变换并加权
        t = self.trend_transform(trend) * batch_weights[:, 0].unsqueeze(1)
        s = self.seasonal_transform(seasonal) * batch_weights[:, 1].unsqueeze(1)
        r = self.residual_transform(residual) * batch_weights[:, 2].unsqueeze(1)
        
        # 融合
        fused = self.fusion_layer(t + s + r)
        
        return fused


class PositionalEncoding(nn.Module):
    """频率可调的位置编码"""
    
    def __init__(self, d_model, max_len=5000, freq_factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.freq_factor = freq_factor
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) * freq_factor
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """添加位置编码到输入特征"""
        return x + self.pe[:, :x.size(1), :]

3.目前的实验结果：
采用DSsoft作为集成策略，从指标上对比基模型是有效果的
Model	1~32		65~96	
    MSE	MAE	MSE	MAE
TimeMoE-50M	0.2983	0.3402	0.3985	0.4113
Near-head	0.2821	0.3497	0.5396	0.5478
Avg-head	0.3001	0.3464	0.4173	0.4265
Far-head	0.3491	0.4235	0.3682	0.4129
集成	0.2753	0.3423	0.3634	0.4077
从特征方面：特征分解后的Short和Long特征与原始特征的相似度分别为-0.198和-0.218，欧氏距离分别为30.04和30.27，表明分解后的特征与原始特征存在显著差异，成功实现了特征的特化。Short和Long特征之间的相似度高达0.966，欧氏距离为4.90，表明两者之间高度相似，这可能限制了专家头的特化效果，导致专家在不同特征上的区分度不够明显。

