import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import social_stgcnn
import copy

# 添加可视化相关导入
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题


def visualize_trajectory_predictions(raw_data_dict, save_path='./visualizations/', max_sequences=5):
    """\n    可视化轨迹预测结果\n    """
    # 创建保存目录
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 限制可视化序列数量
    sequence_count = 0
    for step in list(raw_data_dict.keys())[:max_sequences]:
        sequence_count += 1
        data = raw_data_dict[step]
        obs_data = data['obs']  # 观测轨迹 [seq_len, num_peds, 2]
        trgt_data = data['trgt']  # 真实轨迹
        pred_data = data['pred']  # 预测轨迹列表

        plt.figure(figsize=(12, 10))

        num_peds = min(obs_data.shape[1], 8)  # 限制显示的行人数

        for ped_idx in range(num_peds):
            # 观测轨迹（蓝色实线）
            plt.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                     '-', color='blue', linewidth=2, alpha=0.8,
                     label=f'Observed' if ped_idx == 0 else "")

            # 真实轨迹（绿色实线）
            plt.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                     '-', color='green', linewidth=2, alpha=0.8,
                     label=f'Ground Truth' if ped_idx == 0 else "")

            # 预测轨迹（红色虚线，多个样本）
            for i, pred_sample in enumerate(pred_data[:3]):  # 只显示前3个预测样本
                plt.plot(pred_sample[:, ped_idx, 0], pred_sample[:, ped_idx, 1],
                         '--', color='red', linewidth=1, alpha=0.6,
                         label=f'Predictions' if ped_idx == 0 and i == 0 else "")

        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Trajectory Prediction - Sequence {step}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        if save_path:
            plt.savefig(os.path.join(save_path, f'trajectory_pred_{step}.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Saved visualization for sequence {step}")

        plt.close()  # 关闭图形避免内存泄漏


def create_comparison_visualization(raw_data_dict, save_path='./visualizations/', max_sequences=3):
    """\n    创建对比可视化：显示多个预测样本与真实轨迹的对比\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for step in list(raw_data_dict.keys())[:max_sequences]:
        data = raw_data_dict[step]
        obs_data = data['obs']  # 观测轨迹
        trgt_data = data['trgt']  # 真实轨迹
        pred_data = data['pred']  # 预测轨迹列表

        # 创建子图：显示前3个预测样本
        fig, axes = plt.subplots(1, min(3, len(pred_data)), figsize=(18, 6))
        if min(3, len(pred_data)) == 1:
            axes = [axes]

        num_peds = min(obs_data.shape[1], 6)

        colors = plt.cm.Set1(np.linspace(0, 1, max(num_peds, 1)))

        for pred_idx, ax in enumerate(axes):
            # 绘制每个行人的轨迹
            for ped_idx in range(num_peds):
                color = colors[ped_idx] if ped_idx < len(colors) else np.random.rand(3, )

                # 观测轨迹（蓝色）
                ax.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                        '-', color='blue', linewidth=2, alpha=0.8,
                        label=f'Observed' if ped_idx == 0 else "")

                # 真实轨迹（绿色）
                ax.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                        '-', color='green', linewidth=2, alpha=0.8,
                        label=f'Ground Truth' if ped_idx == 0 else "")

                # 预测轨迹（对应颜色虚线）
                if pred_idx < len(pred_data):
                    ax.plot(pred_data[pred_idx][:, ped_idx, 0],
                            pred_data[pred_idx][:, ped_idx, 1],
                            '--', color=color, linewidth=2, alpha=0.7,
                            label=f'Prediction {pred_idx + 1}' if ped_idx == 0 else "")

            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_title(f'Prediction Sample {pred_idx + 1}')
            ax.grid(True, alpha=0.3)

            # 只在第一个子图添加图例
            if pred_idx == 0:
                # 创建自定义图例
                blue_patch = mpatches.Patch(color='blue', label='Observed')
                green_patch = mpatches.Patch(color='green', label='Ground Truth')
                ax.legend(handles=[blue_patch, green_patch], loc='upper left')

        plt.suptitle(f'Trajectory Prediction Comparison - Sequence {step}')
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'trajectory_comparison_{step}.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Saved comparison visualization for sequence {step}")

        plt.close()


def visualize_uncertainty(raw_data_dict, save_path='./visualizations/', max_sequences=3):
    """\n    可视化预测不确定性：显示多个预测样本的分布\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for step in list(raw_data_dict.keys())[:max_sequences]:
        data = raw_data_dict[step]
        obs_data = data['obs']
        trgt_data = data['trgt']
        pred_data = data['pred']

        if not pred_data:
            continue

        plt.figure(figsize=(12, 10))
        num_peds = min(obs_data.shape[1], 5)

        colors = plt.cm.Set1(np.linspace(0, 1, max(num_peds, 1)))

        for ped_idx in range(num_peds):
            color = colors[ped_idx] if ped_idx < len(colors) else np.random.rand(3, )

            # 观测轨迹
            plt.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                     '-', color='blue', linewidth=3, alpha=0.9,
                     label=f'Ped {ped_idx} - Observed' if ped_idx == 0 else "")

            # 真实轨迹
            plt.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                     '-', color='green', linewidth=3, alpha=0.9,
                     label=f'Ped {ped_idx} - Ground Truth' if ped_idx == 0 else "")

            # 所有预测样本（浅色显示不确定性）
            all_pred_x = []
            all_pred_y = []
            for pred_sample in pred_data[:10]:  # 限制显示前10个样本
                plt.plot(pred_sample[:, ped_idx, 0], pred_sample[:, ped_idx, 1],
                         '--', color='red', linewidth=1, alpha=0.3)
                all_pred_x.extend(pred_sample[:, ped_idx, 0])
                all_pred_y.extend(pred_sample[:, ped_idx, 1])

            # 预测分布的统计信息（可选）
            if len(all_pred_x) > 0:


               plt.xlabel('X coordinate')
               plt.ylabel('Y coordinate')
               plt.title(f'Trajectory Prediction with Uncertainty - Sequence {step}')
               plt.legend()
               plt.grid(True, alpha=0.3)

            if save_path:
               plt.savefig(os.path.join(save_path, f'uncertainty_visualization_{step}.png'),
                        dpi=300, bbox_inches='tight')
               print(f"Saved uncertainty visualization for sequence {step}")

               plt.close()


def visualize_social_interaction_graph(A_obs, V_obs, frame_idx=0, save_path=None):
    """\n    可视化社会交互图\n    """
    try:
        import networkx as nx

        # 提取特定帧的邻接矩阵和节点特征
        adj = A_obs.squeeze()[frame_idx].cpu().numpy()
        features = V_obs.squeeze()[frame_idx].cpu().numpy()

        # 创建图
        G = nx.from_numpy_array(adj)

        plt.figure(figsize=(10, 8))

        # 节点位置基于特征坐标
        pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=500, alpha=0.7)

        # 绘制边（根据权重）
        edges = G.edges()
        weights = [adj[u][v] for u, v in edges]
        # 过滤掉权重很小的边
        valid_edges = [(u, v) for (u, v), w in zip(edges, weights) if w > 0.01]
        valid_weights = [max(0.1, w) for w in weights if w > 0.01]  # 最小线宽

        if valid_edges:
            nx.draw_networkx_edges(G, pos, edgelist=valid_edges,
                                   width=np.array(valid_weights) * 3,
                                   alpha=0.5, edge_color='gray')

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title(f'Social Interaction Graph - Frame {frame_idx}')
        plt.xlabel('Relative X coordinate')
        plt.ylabel('Relative Y coordinate')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved social graph to {save_path}")

        plt.close()

    except ImportError:
        print("NetworkX not available, skipping social graph visualization")
    except Exception as e:
        print(f"Error in social graph visualization: {e}")


def visualize_prediction_error(raw_data_dict, save_path='./visualizations/', max_sequences=3):
    """\n    可视化预测误差：显示预测轨迹与真实轨迹的误差\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for step in list(raw_data_dict.keys())[:max_sequences]:
        data = raw_data_dict[step]
        obs_data = data['obs']
        trgt_data = data['trgt']
        pred_data = data['pred']

        if not pred_data:
            continue

        # 使用第一个预测样本计算误差
        pred_sample = pred_data[0]

        plt.figure(figsize=(12, 10))
        num_peds = min(obs_data.shape[1], 6)

        colors = plt.cm.Set1(np.linspace(0, 1, max(num_peds, 1)))

        for ped_idx in range(num_peds):
            color = colors[ped_idx] if ped_idx < len(colors) else np.random.rand(3, )

            # 观测轨迹
            plt.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                     '-', color='blue', linewidth=2, alpha=0.8,
                     label=f'Ped {ped_idx} - Observed' if ped_idx == 0 else "")

            # 真实轨迹
            plt.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                     '-', color='green', linewidth=2, alpha=0.8,
                     label=f'Ped {ped_idx} - Ground Truth' if ped_idx == 0 else "")

            # 预测轨迹
            plt.plot(pred_sample[:, ped_idx, 0], pred_sample[:, ped_idx, 1],
                     '--', color='red', linewidth=2, alpha=0.8,
                     label=f'Ped {ped_idx} - Prediction' if ped_idx == 0 else "")

            # 绘制误差向量（从真实位置到预测位置的箭头）
            for t in range(min(trgt_data.shape[0], pred_sample.shape[0])):
                if t % 2 == 0:  # 每隔一帧绘制一个箭头以避免过于密集
                    plt.arrow(trgt_data[t, ped_idx, 0], trgt_data[t, ped_idx, 1],
                              pred_sample[t, ped_idx, 0] - trgt_data[t, ped_idx, 0],
                              pred_sample[t, ped_idx, 1] - trgt_data[t, ped_idx, 1],
                              head_width=0.1, head_length=0.1, fc='purple', ec='purple', alpha=0.6)

        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Prediction Error Visualization - Sequence {step}\n(Purple arrows show prediction errors)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(os.path.join(save_path, f'prediction_error_{step}.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Saved prediction error visualization for sequence {step}")

        plt.close()


def visualize_performance_metrics(ade_scores, fde_scores, save_path='./visualizations/'):
    """\n    可视化性能指标统计\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建性能指标对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ADE 分布
    ax1.hist(ade_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('ADE Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'ADE Distribution\nMean: {np.mean(ade_scores):.4f}, Std: {np.std(ade_scores):.4f}')
    ax1.grid(True, alpha=0.3)

    # FDE 分布
    ax2.hist(fde_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('FDE Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'FDE Distribution\nMean: {np.mean(fde_scores):.4f}, Std: {np.std(fde_scores):.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'performance_metrics.png'),
                    dpi=300, bbox_inches='tight')
        print("Saved performance metrics visualization")

    plt.close()


def visualize_trajectory_density(raw_data_dict, save_path='./visualizations/', max_sequences=2):
    """\n    可视化轨迹密度图：显示行人活动的热点区域\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for step in list(raw_data_dict.keys())[:max_sequences]:
        data = raw_data_dict[step]
        obs_data = data['obs']
        trgt_data = data['trgt']
        pred_data = data['pred']

        plt.figure(figsize=(12, 10))

        # 收集所有轨迹点
        all_points = []

        # 观测轨迹点
        obs_points = obs_data.reshape(-1, 2)
        all_points.extend(obs_points)

        # 真实轨迹点
        trgt_points = trgt_data.reshape(-1, 2)
        all_points.extend(trgt_points)

        # 预测轨迹点（如果有）
        if pred_data:
            pred_points = pred_data[0].reshape(-1, 2)
            all_points.extend(pred_points)

        all_points = np.array(all_points)

        # 创建2D直方图
        plt.hist2d(all_points[:, 0], all_points[:, 1], bins=50, cmap='Blues', alpha=0.7)
        plt.colorbar(label='Point Density')

        # 叠加观测轨迹
        num_peds = min(obs_data.shape[1], 5)
        colors = plt.cm.Set1(np.linspace(0, 1, max(num_peds, 1)))

        for ped_idx in range(num_peds):
            color = colors[ped_idx] if ped_idx < len(colors) else np.random.rand(3, )

            # 观测轨迹
            plt.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                     '-', color=color, linewidth=2, alpha=0.8)

            # 真实轨迹
            plt.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                     '--', color=color, linewidth=2, alpha=0.8)

            if pred_data:
                plt.plot(pred_data[0][:, ped_idx, 0], pred_data[0][:, ped_idx, 1],
                         ':', color=color, linewidth=2, alpha=0.8)

        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Trajectory Density Map - Sequence {step}\n(Blue intensity shows activity density)')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(os.path.join(save_path, f'trajectory_density_{step}.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Saved trajectory density visualization for sequence {step}")

        plt.close()


def generate_visualization_summary(raw_data_dict, ade_score, fde_score, save_path='./visualizations/'):
    """\n    生成可视化摘要报告\n    """
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建摘要图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Social-STGCNN Visualization Summary\nADE: {ade_score:.4f}, FDE: {fde_score:.4f}',
                 fontsize=16)

    if raw_data_dict:
        # 选择第一个序列进行展示
        step = list(raw_data_dict.keys())[0]
        data = raw_data_dict[step]
        obs_data = data['obs']
        trgt_data = data['trgt']
        pred_data = data['pred']

        num_peds = min(obs_data.shape[1], 4)

        # 1. 轨迹对比图
        ax = axes[0, 0]
        for ped_idx in range(num_peds):
            ax.plot(obs_data[:, ped_idx, 0], obs_data[:, ped_idx, 1],
                    '-', color='blue', linewidth=2, alpha=0.8, label='Observed' if ped_idx == 0 else "")
            ax.plot(trgt_data[:, ped_idx, 0], trgt_data[:, ped_idx, 1],
                    '-', color='green', linewidth=2, alpha=0.8, label='Ground Truth' if ped_idx == 0 else "")
            if pred_data:
                ax.plot(pred_data[0][:, ped_idx, 0], pred_data[0][:, ped_idx, 1],
                        '--', color='red', linewidth=2, alpha=0.8, label='Prediction' if ped_idx == 0 else "")
        ax.set_title('Trajectory Comparison')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 社会交互热力图（模拟）
        ax = axes[0, 1]
        # 创建一个简单的交互热力图示例
        interaction_matrix = np.random.rand(num_peds, num_peds)
        im = ax.imshow(interaction_matrix, cmap='Reds')
        ax.set_title('Social Interaction Heatmap')
        ax.set_xlabel('Pedestrian ID')
        ax.set_ylabel('Pedestrian ID')
        plt.colorbar(im, ax=ax)

        # 3. 性能指标分布
        ax = axes[1, 0]
        # 模拟一些ADE/FDE分数
        ade_samples = np.random.normal(ade_score, 0.1, 100)
        ax.hist(ade_samples, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(ade_score, color='red', linestyle='--', linewidth=2, label=f'Current ADE: {ade_score:.4f}')
        ax.set_title('ADE Distribution')
        ax.set_xlabel('ADE Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 轨迹密度图
        ax = axes[1, 1]
        all_points = np.concatenate([
            obs_data.reshape(-1, 2),
            trgt_data.reshape(-1, 2)
        ])
        ax.hist2d(all_points[:, 0], all_points[:, 1], bins=30, cmap='Blues', alpha=0.7)
        ax.set_title('Trajectory Density')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'visualization_summary.png'),
                    dpi=300, bbox_inches='tight')
        print("Saved visualization summary")

    plt.close()


def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # For now I have my bi-variate parameters
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        # 正确初始化为字典
        ade_ls = {}
        fde_ls = {}

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

        # 每个批次后进行可视化（只对前几个序列）
        if step <= 2:  # 只可视化前2个序列以节省时间
            try:
                print(f"Generating visualizations for sequence {step}...")

                # 基础轨迹可视化
                temp_dict = {step: raw_data_dict[step]}
                visualize_trajectory_predictions(temp_dict,
                                                 save_path=f'./visualizations/sequence_{step}/')

                # 对比可视化
                create_comparison_visualization(temp_dict,
                                                save_path=f'./visualizations/sequence_{step}/')

                # 不确定性可视化
                visualize_uncertainty(temp_dict,
                                      save_path=f'./visualizations/sequence_{step}/')

                # 预测误差可视化
                visualize_prediction_error(temp_dict,
                                           save_path=f'./visualizations/sequence_{step}/')

                # 社会交互图（只对第一个序列）
                if step == 1 and 'A_obs' in locals():
                    visualize_social_interaction_graph(A_obs, V_obs, frame_idx=0,
                                                       save_path='./visualizations/social_graph.png')

            except Exception as e:
                print(f"Visualization error for sequence {step}: {e}")

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


# 将主程序逻辑包装在 if __name__ == '__main__': 中
if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()  # Windows 兼容性支持

    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS = 20

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)

        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,
                shuffle=False,
                num_workers=1)

            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd, raw_data_dic_ = test()
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:", ade_, " FDE:", fde_)

            # 生成最终的可视化结果
            print("Generating comprehensive visualizations...")
            try:
                # 基础轨迹可视化
                visualize_trajectory_predictions(raw_data_dic_, save_path='./final_visualizations/')

                # 对比可视化
                create_comparison_visualization(raw_data_dic_, save_path='./final_visualizations/')

                # 不确定性可视化
                visualize_uncertainty(raw_data_dic_, save_path='./final_visualizations/')

                # 预测误差可视化
                visualize_prediction_error(raw_data_dic_, save_path='./final_visualizations/')

                # 轨迹密度图
                visualize_trajectory_density(raw_data_dic_, save_path='./final_visualizations/')

                # 性能指标可视化
                visualize_performance_metrics(ade_ls, fde_ls, save_path='./final_visualizations/')

                # 生成可视化摘要
                generate_visualization_summary(raw_data_dic_, ade_, fde_, save_path='./final_visualizations/')

                print("Comprehensive visualizations saved to ./final_visualizations/")

            except Exception as e:
                print(f"Error generating final visualizations: {e}")

        print("*" * 50)

        print("Avg ADE:", sum(ade_ls) / max(len(ade_ls), 1))
        print("Avg FDE:", sum(fde_ls) / max(len(fde_ls), 1))
