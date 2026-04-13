#!/usr/bin/env python3
"""
训练通知模块
============

支持飞书机器人 Webhook 通知训练进度。
"""

import json
import logging
import os
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeishuNotifier:
    """飞书机器人通知器"""
    
    def __init__(self, webhook_url: Optional[str] = None, enabled: bool = True):
        """
        初始化飞书通知器
        
        Args:
            webhook_url: 飞书机器人 Webhook URL，可从环境变量 FEISHU_WEBHOOK_URL 读取
            enabled: 是否启用通知
        """
        self.webhook_url = webhook_url or os.environ.get('FEISHU_WEBHOOK_URL', '')
        self.enabled = enabled and bool(self.webhook_url)
        
        if self.enabled:
            logger.info(f"📢 FeishuNotifier enabled (webhook: ...{self.webhook_url[-20:]})")
        else:
            logger.info("📢 FeishuNotifier disabled (no webhook_url)")
    
    def _send_message(self, content: Dict) -> bool:
        """发送消息到飞书"""
        if not self.enabled:
            return False
        
        try:
            import urllib.request
            import urllib.error
            
            data = json.dumps(content).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('code') == 0 or result.get('StatusCode') == 0:
                    return True
                else:
                    logger.warning(f"Feishu API error: {result}")
                    return False
                    
        except Exception as e:
            logger.warning(f"Failed to send Feishu notification: {e}")
            return False
    
    def send_epoch_report(
        self,
        epoch: int,
        total_epochs: int,
        stage_name: str,
        stage_idx: int,
        total_stages: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        eta: str,
        epoch_time: str,
        is_best: bool = False,
        best_val_loss: Optional[float] = None,
    ) -> bool:
        """发送 epoch 训练报告"""
        now = datetime.now().strftime('%H:%M:%S')
        
        # 状态标识
        status_emoji = "⭐" if is_best else "📊"
        
        # 提取指标 - 使用新术语
        train_loss = train_metrics.get('total_loss', 0)
        train_hm = train_metrics.get('heatmap_loss', 0)
        train_traj = train_metrics.get('action_loss', 0)  # trajectory
        train_prog = train_metrics.get('stop_loss', 0)    # progress
        
        val_loss = val_metrics.get('val_loss', 0)
        val_hm = val_metrics.get('val_heatmap_loss', 0)
        val_traj = val_metrics.get('val_action_loss', 0)
        val_prog = val_metrics.get('val_stop_loss', 0)
        
        # 进度条
        progress_pct = epoch / total_epochs * 100
        progress_bar = "█" * int(progress_pct / 10) + "░" * (10 - int(progress_pct / 10))
        
        # 构建消息内容
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{status_emoji} Epoch {epoch}/{total_epochs} {'🎉 Best!' if is_best else ''}"
                    },
                    "template": "green" if is_best else "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**{stage_name}** | {now} | ⏱️ {epoch_time} | ETA: {eta}\n"
                                       f"[{progress_bar}] {progress_pct:.0f}%"
                        }
                    },
                    {"tag": "hr"},
                    {
                        "tag": "div",
                        "fields": [
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**Train** `{train_loss:.4f}`\n"
                                               f"hm: {train_hm:.4f}\n"
                                               f"traj: {train_traj:.4f}\n"
                                               f"prog: {train_prog:.4f}"
                                }
                            },
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**Val** `{val_loss:.4f}`\n"
                                               f"hm: {val_hm:.4f}\n"
                                               f"traj: {val_traj:.4f}\n"
                                               f"prog: {val_prog:.4f}"
                                }
                            }
                        ]
                    }
                ]
            }
        }
        
        # 添加最佳模型标记
        if is_best and best_val_loss is not None:
            content["card"]["elements"].append({
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"🏆 Best val_loss: {best_val_loss:.4f}"
                    }
                ]
            })
        
        return self._send_message(content)
    
    def send_training_start(
        self,
        config_name: str,
        stages: list,
        total_epochs: int,
        extra_info: Optional[Dict] = None,
    ) -> bool:
        """发送训练开始通知"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        stage_info = ""
        for s in stages:
            name = s.get('name', 'unknown')
            epochs = s.get('epochs', '?')
            stage_info += f"• {name} ({epochs} epochs)\n"
        
        # 额外信息
        extra_text = ""
        if extra_info:
            if 'batch_size' in extra_info:
                extra_text += f"**Batch**: {extra_info['batch_size']}\n"
            if 'gpu' in extra_info:
                extra_text += f"**GPU**: {extra_info['gpu']}\n"
            if 'dataset_size' in extra_info:
                extra_text += f"**Samples**: {extra_info['dataset_size']}\n"
        
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "🚀 训练开始"
                    },
                    "template": "wathet"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Config**: `{config_name}`\n"
                                       f"**Time**: {now}\n"
                                       f"**Epochs**: {total_epochs}\n"
                                       f"{extra_text}"
                                       f"\n**Stages**:\n{stage_info}"
                        }
                    }
                ]
            }
        }
        
        return self._send_message(content)
    
    def send_training_complete(
        self,
        total_time: str,
        best_val_loss: float,
        final_stage: str,
        final_epoch: int = None,
    ) -> bool:
        """发送训练完成通知"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        epoch_info = f" (Epoch {final_epoch})" if final_epoch else ""
        
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "✅ 训练完成"
                    },
                    "template": "green"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Time**: {now}\n"
                                       f"**Duration**: {total_time}\n"
                                       f"**Best val_loss**: `{best_val_loss:.4f}`{epoch_info}\n"
                                       f"**Stage**: {final_stage}"
                        }
                    }
                ]
            }
        }
        
        return self._send_message(content)
    
    def send_training_error(self, error_msg: str, stage: str = None, epoch: int = None) -> bool:
        """发送训练错误通知"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        location = stage or 'Unknown'
        if epoch:
            location += f" (Epoch {epoch})"
        
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "❌ 训练错误"
                    },
                    "template": "red"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**Time**: {now}\n"
                                       f"**Location**: {location}\n"
                                       f"**Error**:\n```\n{error_msg[:500]}\n```"
                        }
                    }
                ]
            }
        }
        
        return self._send_message(content)
    
    def send_checkpoint_saved(self, epoch: int, val_loss: float, is_best: bool = False) -> bool:
        """发送检查点保存通知（可选，避免太多通知可以不调用）"""
        if not is_best:
            return True  # 只通知 best 模型
        
        content = {
            "msg_type": "text",
            "content": {
                "text": f"💾 Best model saved: Epoch {epoch}, val_loss={val_loss:.4f}"
            }
        }
        
        return self._send_message(content)


def create_notifier(cfg: Dict) -> Optional[FeishuNotifier]:
    """根据配置创建通知器"""
    notify_cfg = cfg.get('log', {}).get('notify', {})
    
    if not notify_cfg.get('enabled', False):
        return None
    
    platform = notify_cfg.get('platform', 'feishu')
    
    if platform == 'feishu':
        webhook_url = notify_cfg.get('webhook_url', '')
        # 支持环境变量替换
        if webhook_url.startswith('${') and webhook_url.endswith('}'):
            env_var = webhook_url[2:-1]
            webhook_url = os.environ.get(env_var, '')
        
        return FeishuNotifier(webhook_url=webhook_url, enabled=True)
    
    else:
        logger.warning(f"Unsupported notification platform: {platform}")
        return None
