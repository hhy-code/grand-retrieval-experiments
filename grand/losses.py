"""Ranking and knowledge-distillation losses used by GRAND."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ranking_loss(positive, negative):
    """Paper Eq. 1 with its stated unit margin and one sampled negative."""
    return F.relu(1.0 - positive + negative)


def score_distillation(student_positive, student_negative, teacher_positive, teacher_negative, temperature):
    student = torch.stack((student_positive, student_negative)).unsqueeze(0) / temperature
    teacher = torch.stack((teacher_positive, teacher_negative)).unsqueeze(0) / temperature
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher.detach(), dim=-1), reduction="batchmean")


def node_distillation(student_query, student_candidate, teacher_query, teacher_candidate, temperature):
    """Paper Eq. 11-12: align candidate-node attention per query node."""
    student_logits = student_query @ student_candidate.t() / temperature
    teacher_logits = teacher_query.detach() @ teacher_candidate.detach().t() / temperature
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="sum")


def _component_means(nodes, edge_index):
    """Pool connected components when no explicit partition is available."""
    neighbors = [[] for _ in range(nodes.shape[0])]
    for source, target in edge_index.t().tolist():
        neighbors[source].append(target)
        neighbors[target].append(source)
    remaining, groups = set(range(nodes.shape[0])), []
    while remaining:
        start = remaining.pop()
        component, stack = [start], [start]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        indexes = torch.tensor(component, device=nodes.device)
        groups.append(nodes[indexes].mean(dim=0))
    return torch.stack(groups)


def subgraph_distillation(student_query, student_candidate, teacher_query, teacher_candidate, query_edge, candidate_edge, temperature, query_groups=None, candidate_groups=None):
    """Paper Eq. 14-16: align subgraph attention distributions."""
    def pool(nodes, edge_index, groups):
        if groups is None:
            return _component_means(nodes, edge_index)
        return torch.stack([nodes[indexes].sum(dim=0) for indexes in groups])

    student_query_groups = pool(student_query, query_edge, query_groups)
    student_candidate_groups = pool(student_candidate, candidate_edge, candidate_groups)
    teacher_query_groups = pool(teacher_query.detach(), query_edge, query_groups)
    teacher_candidate_groups = pool(teacher_candidate.detach(), candidate_edge, candidate_groups)
    student_logits = student_query_groups @ student_candidate_groups.t() / temperature
    teacher_logits = teacher_query_groups @ teacher_candidate_groups.t() / temperature
    return F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="sum")
