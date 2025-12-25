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
    
    def __init__(self, webhook_url: str = None, enabled: bool = True):
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
        """发送消息到飞书
        
        Args:
            content: 消息内容（飞书卡片格式）
            
        Returns:
            是否发送成功
        """
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
        best_val_loss: float = None,
    ) -> bool:
        """
        发送 epoch 训练报告
        
        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
            stage_name: 阶段名称
            stage_idx: 阶段索引
            total_stages: 总阶段数
            train_metrics: 训练指标
            val_metrics: 验证指标
            eta: 预估剩余时间
            epoch_time: 本 epoch 用时
            is_best: 是否是最佳模型
            best_val_loss: 历史最佳验证损失
            
        Returns:
            是否发送成功
        """
        # 构建飞书卡片消息
        now = datetime.now().strftime('%H:%M:%S')
        
        # 状态标识
        status_emoji = "⭐" if is_best else "✅"
        status_text = "🎉 新最佳模型！" if is_best else ""
        
        # 提取指标
        train_loss = train_metrics.get('total_loss', 0)
        train_hm = train_metrics.get('heatmap_loss', 0)
        train_act = train_metrics.get('action_loss', 0)
        val_loss = val_metrics.get('val_loss', 0)
        val_hm = val_metrics.get('val_heatmap_loss', 0)
        val_act = val_metrics.get('val_action_loss', 0)
        
        # 构建消息内容
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{status_emoji} VLN 训练进度 - Epoch {epoch}/{total_epochs}"
                    },
                    "template": "green" if is_best else "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**阶段**: {stage_name} ({stage_idx+1}/{total_stages})\n"
                                       f"**时间**: {now} | 用时: {epoch_time} | ETA: {eta}"
                        }
                    },
                    {
                        "tag": "hr"
                    },
                    {
                        "tag": "div",
                        "fields": [
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**📊 Train Loss**\n{train_loss:.4f}\n(hm: {train_hm:.4f}, act: {train_act:.4f})"
                                }
                            },
                            {
                                "is_short": True,
                                "text": {
                                    "tag": "lark_md",
                                    "content": f"**📈 Val Loss**\n{val_loss:.4f}\n(hm: {val_hm:.4f}, act: {val_act:.4f})"
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
        elif status_text:
            content["card"]["elements"].append({
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": status_text
                    }
                ]
            })
        
        return self._send_message(content)
    
    def send_training_start(
        self,
        config_name: str,
        stages: list,
        total_epochs: int,
    ) -> bool:
        """发送训练开始通知"""
        stage_info = "\n".join([f"  - {s['name']} ({s['epochs']} epochs)" for s in stages])
        
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "🚀 VLN 训练开始"
                    },
                    "template": "wathet"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**配置**: {config_name}\n"
                                       f"**总 Epochs**: {total_epochs}\n"
                                       f"**训练阶段**:\n{stage_info}"
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
    ) -> bool:
        """发送训练完成通知"""
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "✅ VLN 训练完成"
                    },
                    "template": "green"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**总用时**: {total_time}\n"
                                       f"**最佳 val_loss**: {best_val_loss:.4f}\n"
                                       f"**最终阶段**: {final_stage}"
                        }
                    }
                ]
            }
        }
        
        return self._send_message(content)
    
    def send_training_error(self, error_msg: str, stage: str = None) -> bool:
        """发送训练错误通知"""
        content = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "❌ VLN 训练错误"
                    },
                    "template": "red"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**阶段**: {stage or 'Unknown'}\n"
                                       f"**错误**: {error_msg[:500]}"  # 限制长度
                        }
                    }
                ]
            }
        }
        
        return self._send_message(content)


def create_notifier(cfg: Dict) -> Optional[FeishuNotifier]:
    """
    根据配置创建通知器
    
    Args:
        cfg: 完整配置字典
        
    Returns:
        FeishuNotifier 或 None
    """
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

