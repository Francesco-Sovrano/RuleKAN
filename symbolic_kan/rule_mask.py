import math
from typing import Optional, Union

import torch
import torch.nn as nn


class RuleMaskProduct(nn.Module):
    """Learn sparse products of KAN edge functions with categorical factor slots.

    This is a numerically/optimisation-stable replacement for independent
    Bernoulli gates.  Each output rule owns ``max_order`` factor *slots*.  A slot
    chooses exactly one input edge function, or (for optional slots) the
    multiplicative identity ``1``.  Hence both the hard forward model and the
    soft gradient surrogate contain at most ``max_order`` multiplicative
    factors.

    For rule ``r`` and slot ``s`` we learn a categorical distribution over
    ``in_dim`` candidate KAN edge functions plus an identity candidate.  The
    first ``min_order`` slots cannot choose identity.  Hard choices are used in
    the numerical forward pass, while a soft categorical mixture supplies
    gradients to the choice logits and unselected edge functions.

    This avoids the major failure mode of the old implementation where the hard
    forward pass was top-k constrained but the soft backward pass still
    multiplied *all* input edges.  That mismatch can drive mask probabilities
    and KAN edge magnitudes to pathological values.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_prob: float = 0.25,
        temperature: float = 1.0,
        threshold: float = 0.5,
        stochastic: bool = False,
        min_order: int = 1,
        max_order: Optional[int] = None,
        target_order: Optional[Union[int, float]] = None,
        surrogate_clip: Optional[float] = 8.0,
        rule_orders: Optional[list] = None,
        gate_mode: str = "hard_st",
        asinh_scale: float = 4.0,
        learnable_asinh_scale: bool = True,
    ):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        if not (0.0 < init_prob < 1.0):
            raise ValueError("init_prob must be strictly between 0 and 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be strictly between 0 and 1")
        if min_order < 0:
            raise ValueError("min_order must be >= 0")
        if max_order is not None and max_order < 1:
            raise ValueError("max_order must be >= 1 or None")
        if max_order is not None and min_order > max_order:
            raise ValueError("min_order cannot exceed max_order")
        if surrogate_clip is not None and surrogate_clip <= 0:
            raise ValueError("surrogate_clip must be positive or None")
        if gate_mode not in {"hard_st", "gmp"}:
            raise ValueError("gate_mode must be 'hard_st' or 'gmp'")
        if asinh_scale <= 0:
            raise ValueError("asinh_scale must be positive")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.min_order = min(int(min_order), self.in_dim)
        self.max_order = self.in_dim if max_order is None else min(int(max_order), self.in_dim)
        self.num_slots = self.max_order
        self.temperature = float(temperature)
        self.threshold = float(threshold)  # kept for backwards-compatible API
        self.stochastic = bool(stochastic)
        self.target_order = None if target_order is None else float(target_order)
        self.surrogate_clip = None if surrogate_clip is None else float(surrogate_clip)
        self.gate_mode = str(gate_mode)
        self.learnable_asinh_scale = bool(learnable_asinh_scale)

        if rule_orders is not None:
            if len(rule_orders) != self.out_dim:
                raise ValueError(f"rule_orders must have length {self.out_dim}")
            orders = torch.as_tensor(rule_orders, dtype=torch.long)
            if ((orders < 0) | (orders > self.max_order)).any():
                raise ValueError(f"rule_orders entries must be in [0,{self.max_order}]")
            self.register_buffer("rule_orders", orders)
        else:
            self.rule_orders = None

        # logits: [rule, slot, feature + identity]
        # Optional slots start with total probability init_prob on real features
        # and 1-init_prob on identity. Mandatory slots ignore identity.
        feat_logit = math.log(init_prob / self.in_dim)
        id_logit = math.log(1.0 - init_prob)
        logits = torch.full(
            (self.out_dim, self.num_slots, self.in_dim + 1),
            feat_logit,
        )
        logits[..., self.in_dim] = id_logit

        # Mandatory slots should start as ordinary categorical feature choices.
        if self.min_order > 0:
            logits[:, : self.min_order, : self.in_dim] = -math.log(self.in_dim)
            logits[:, : self.min_order, self.in_dim] = -20.0

        self.logits = nn.Parameter(logits)
        with torch.no_grad():
            self.logits[..., : self.in_dim].add_(0.03 * torch.randn_like(self.logits[..., : self.in_dim]))

        # GMP-style candidate shortlist.  Candidates are removed gradually by
        # top-k pruning rather than forcing an early argmax.  The buffer is
        # serialized and sliced during KAN pruning.
        self.register_buffer(
            "candidate_mask",
            torch.ones(self.out_dim, self.num_slots, self.in_dim + 1, dtype=torch.bool),
        )
        self.register_buffer("discretized", torch.tensor(False, dtype=torch.bool))

        # Scaled-asinh variance compression from the in-context SR paper.  We
        # use it only on the gradient path, while keeping the numerical forward
        # value equal to the uncompressed soft mixture.  This damps outliers
        # without changing the function represented before discretisation.
        log_s = math.log(float(asinh_scale))
        scale_init = torch.full((self.out_dim, self.num_slots), log_s)
        # Keep the hard-ST optimizer parameter set identical to the proven v3
        # implementation. Even an unused extra Adam parameter can change foreach
        # grouping/roundoff enough to flip a discrete argmax mask in this brittle
        # optimization. The asinh scale is only relevant to GMP, so hard-ST stores
        # it as a buffer rather than a trainable Parameter.
        if self.gate_mode == "gmp":
            self.log_asinh_scale = nn.Parameter(
                scale_init,
                requires_grad=self.learnable_asinh_scale,
            )
        else:
            self.register_buffer("log_asinh_scale", scale_init)

    # ------------------------------------------------------------------
    # probabilities / masks
    # ------------------------------------------------------------------
    def _slot_probabilities(self, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sequential categorical probabilities [O,S,I+1].

        Later slots are differentiably discouraged from re-selecting features
        already claimed by earlier slots. This keeps the soft surrogate close
        to the duplicate-free hard greedy assignment.
        """
        base = (self.logits / max(self.temperature, 1e-6)).clone()
        candidate_mask = self.candidate_mask.to(device=base.device)

        if active_mask is None:
            active = torch.ones((self.out_dim, self.in_dim), dtype=torch.bool, device=base.device)
        else:
            active = active_mask.to(device=base.device).bool()
            if active.shape != (self.out_dim, self.in_dim):
                raise ValueError(
                    f"active_mask must have shape {(self.out_dim, self.in_dim)}, got {tuple(active.shape)}"
                )

        remaining = active.to(base.dtype)
        probs = []
        for s in range(self.num_slots):
            logits_s = base[:, s, :].clone()
            logits_s = logits_s.masked_fill(~candidate_mask[:, s, :], -1e9)
            # Differentiable without-replacement relaxation: a feature that an
            # earlier slot already strongly selected gets a large negative bias.
            logits_s[:, : self.in_dim] = (
                logits_s[:, : self.in_dim]
                + torch.log(remaining.clamp_min(1e-6))
            )
            logits_s[:, : self.in_dim] = logits_s[:, : self.in_dim].masked_fill(~active, -1e9)

            if self.rule_orders is None:
                if s < self.min_order:
                    logits_s[:, self.in_dim] = -1e9
            else:
                # Exact per-rule interaction order. Slots below the requested
                # order must choose a feature; slots above it are identity-only.
                must_feature = self.rule_orders > s
                must_identity = ~must_feature
                logits_s[must_feature, self.in_dim] = -1e9
                logits_s[must_identity, : self.in_dim] = -1e9
                logits_s[must_identity, self.in_dim] = 0.0

            no_active = ~active.any(dim=-1)
            if no_active.any():
                logits_s[no_active, self.in_dim] = 0.0

            p_s = torch.softmax(logits_s, dim=-1)
            probs.append(p_s)
            remaining = remaining * (1.0 - p_s[:, : self.in_dim])

        return torch.stack(probs, dim=1)

    def slot_probabilities(self, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return categorical slot probabilities [out_dim, max_order, in_dim+1]."""
        return self._slot_probabilities(active_mask=active_mask)

    def probabilities(self) -> torch.Tensor:
        """Approximate probability that each feature participates in each rule.

        Returns [out_dim, in_dim].  It is the probability that a feature is
        chosen by at least one slot under the independent soft slot model.
        """
        p = self._slot_probabilities()[..., : self.in_dim]
        return 1.0 - torch.prod(1.0 - p, dim=1)

    def _hard_slots(
        self,
        probs: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        stochastic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Greedy one-hot slot assignments [O,S,I+1], without duplicate features."""
        do_sample = self.stochastic if stochastic is None else bool(stochastic)
        hard = torch.zeros_like(probs)

        if active_mask is None:
            active = torch.ones((self.out_dim, self.in_dim), dtype=torch.bool, device=probs.device)
        else:
            active = active_mask.to(device=probs.device).bool()

        for r in range(self.out_dim):
            used = torch.zeros(self.in_dim, dtype=torch.bool, device=probs.device)
            for s in range(self.num_slots):
                allowed = torch.zeros(self.in_dim + 1, dtype=torch.bool, device=probs.device)
                allowed[: self.in_dim] = active[r] & (~used)
                if self.rule_orders is None:
                    if s >= self.min_order:
                        allowed[self.in_dim] = True
                else:
                    order_r = int(self.rule_orders[r].item())
                    if s >= order_r:
                        allowed[:] = False
                        allowed[self.in_dim] = True

                # Degenerate case: no real choice left in a mandatory slot.
                if not allowed.any():
                    allowed[self.in_dim] = True

                scores = probs[r, s].clone()
                scores = scores.masked_fill(~allowed, 0.0)
                total = scores.sum()
                if total <= 0:
                    idx = self.in_dim
                elif do_sample and self.training:
                    idx = torch.multinomial(scores / total, 1).item()
                else:
                    idx = torch.argmax(scores).item()

                hard[r, s, idx] = 1.0
                if idx < self.in_dim:
                    used[idx] = True

        return hard

    def hard_slots(
        self,
        active_mask: Optional[torch.Tensor] = None,
        stochastic: Optional[bool] = None,
    ) -> torch.Tensor:
        p = self._slot_probabilities(active_mask=active_mask)
        return self._hard_slots(p, active_mask=active_mask, stochastic=stochastic).detach()

    def hard_mask(
        self,
        active_mask: Optional[torch.Tensor] = None,
        stochastic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return a detached binary [O,I] union mask for inspection/extraction."""
        slots = self.hard_slots(active_mask=active_mask, stochastic=stochastic)
        return (slots[..., : self.in_dim].sum(dim=1) > 0).to(slots.dtype).detach()

    def straight_through_mask(
        self,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Backwards-compatible [O,I] hard-union mask with soft union gradients."""
        hard = self.hard_mask(active_mask=active_mask, stochastic=None)
        soft = self.probabilities()
        if active_mask is not None:
            soft = soft * active_mask.to(device=soft.device, dtype=soft.dtype)
        return hard + soft - soft.detach()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        edge_values: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ):
        if edge_values.ndim != 3:
            raise ValueError(
                f"edge_values must have shape [B,O,I], got {tuple(edge_values.shape)}"
            )
        if edge_values.shape[1:] != (self.out_dim, self.in_dim):
            raise ValueError(
                "edge_values shape mismatch: expected "
                f"[B,{self.out_dim},{self.in_dim}], got {tuple(edge_values.shape)}"
            )

        probs = self._slot_probabilities(active_mask=active_mask)
        hard = self._hard_slots(probs, active_mask=active_mask, stochastic=None)

        # Candidate value in the last column is the multiplicative identity.
        ones = torch.ones(
            (*edge_values.shape[:2], 1),
            dtype=edge_values.dtype,
            device=edge_values.device,
        )
        hard_candidates = torch.cat([edge_values, ones], dim=-1)

        hard_cast = hard.to(dtype=edge_values.dtype, device=edge_values.device)
        probs_cast = probs.to(dtype=edge_values.dtype, device=edge_values.device)

        hard_slot_values = torch.einsum("osi,boi->bos", hard_cast, hard_candidates)
        hard_out = torch.prod(hard_slot_values, dim=-1)

        if bool(self.discretized.item()):
            # After GMP discretisation the product is genuinely symbolic: each
            # slot uses one selected edge function (or identity).
            out = hard_out
        elif self.gate_mode == "gmp":
            # GMP-style training: do NOT use hard argmax choices in the forward
            # pass.  Each slot is a differentiable mixture over its current
            # shortlist.  This removes the mask-switching loss spikes produced
            # by the former hard-forward STE training.
            raw_slot_values = torch.einsum("osi,boi->bos", probs_cast, hard_candidates)

            # Scaled-asinh compression on the gradient path.  Forward values
            # remain the raw soft mixture, but gate/KAN gradients see compressed
            # candidates, following the paper's variance-compression idea.
            stable_slots = []
            for s_idx in range(self.num_slots):
                scale = self.log_asinh_scale[:, s_idx].exp().clamp(1e-3, 1e3)
                scale_b = scale.unsqueeze(0).unsqueeze(-1)
                stable_edges = scale_b * torch.asinh(edge_values / scale_b)
                stable_candidates = torch.cat([stable_edges, ones], dim=-1)
                stable_s = torch.einsum(
                    "oi,boi->bo", probs_cast[:, s_idx, :], stable_candidates
                )
                stable_slots.append(stable_s)
            stable_slot_values = torch.stack(stable_slots, dim=-1)

            slot_values = raw_slot_values.detach() + stable_slot_values - stable_slot_values.detach()
            out = torch.prod(slot_values, dim=-1)
        else:
            # Backwards-compatible hard-forward/soft-backward mode.
            if self.surrogate_clip is None:
                soft_edges = edge_values
            else:
                c = self.surrogate_clip
                soft_edges = c * torch.tanh(edge_values / c)
            soft_candidates = torch.cat([soft_edges, ones], dim=-1)
            soft_slot_values = torch.einsum("osi,boi->bos", probs_cast, soft_candidates)
            soft_out = torch.prod(soft_slot_values, dim=-1)
            out = hard_out + soft_out - soft_out.detach()

        # Kill a fully-pruned rule rather than letting all identity slots return 1.
        if active_mask is not None:
            has_active = active_mask.to(device=out.device).bool().any(dim=-1)
            out = out * has_active.to(out.dtype).unsqueeze(0)

        if return_mask:
            union = (hard[..., : self.in_dim].sum(dim=1) > 0).to(edge_values.dtype)
            return out, union.detach()
        return out

    # ------------------------------------------------------------------
    # GMP-style shortlist pruning / diagnostics / discretisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune_candidates_topk(self, k: int) -> None:
        """Keep at most ``k`` real feature candidates per active slot.

        Identity is retained when that slot is allowed to be optional.  This is
        the direct analogue of GMP's periodic top-k operator pruning.
        """
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        probs = self._slot_probabilities()
        for r in range(self.out_dim):
            order_r = None if self.rule_orders is None else int(self.rule_orders[r].item())
            for s_idx in range(self.num_slots):
                if order_r is not None and s_idx >= order_r:
                    self.candidate_mask[r, s_idx, :] = False
                    self.candidate_mask[r, s_idx, self.in_dim] = True
                    continue
                allowed = self.candidate_mask[r, s_idx, : self.in_dim].clone()
                ids = torch.nonzero(allowed, as_tuple=False).squeeze(-1)
                if ids.numel() > k:
                    scores = probs[r, s_idx, ids]
                    keep_local = torch.topk(scores, k=k).indices
                    keep_ids = ids[keep_local]
                    self.candidate_mask[r, s_idx, : self.in_dim] = False
                    self.candidate_mask[r, s_idx, keep_ids] = True
                # identity eligibility is controlled by _slot_probabilities; do
                # not accidentally eliminate it for optional slots.
                self.candidate_mask[r, s_idx, self.in_dim] = True

    def gate_diagnostics(self) -> dict:
        """Return mean normalized entropy, top-two margin and shortlist size."""
        with torch.no_grad():
            p = self._slot_probabilities().clamp_min(1e-12)
            ent = -(p * p.log()).sum(dim=-1)
            allowed = self.candidate_mask.sum(dim=-1).clamp_min(2).to(p.dtype)
            norm_ent = ent / allowed.log()
            top2 = torch.topk(p, k=min(2, p.shape[-1]), dim=-1).values
            if top2.shape[-1] == 1:
                margin = torch.ones_like(top2[..., 0])
            else:
                margin = top2[..., 0] - top2[..., 1]
            real_shortlist = self.candidate_mask[..., : self.in_dim].sum(dim=-1).float()
            return {
                "entropy": float(norm_ent.mean()),
                "margin": float(margin.mean()),
                "shortlist": float(real_shortlist.mean()),
            }

    @torch.no_grad()
    def discretize(self, freeze: bool = True, strength: float = 18.0) -> torch.Tensor:
        """Commit each slot to its current argmax, optionally freezing gates."""
        hard = self.hard_slots(stochastic=False)
        self.logits.fill_(-float(strength))
        for r in range(self.out_dim):
            for s_idx in range(self.num_slots):
                idx = int(torch.argmax(hard[r, s_idx]).item())
                self.logits[r, s_idx, idx] = float(strength)
                self.candidate_mask[r, s_idx, :] = False
                self.candidate_mask[r, s_idx, idx] = True
        self.discretized.fill_(True)
        self.logits.requires_grad_(not freeze)
        return self.hard_mask(stochastic=False)

    @torch.no_grad()
    def undiscretize(self) -> None:
        self.discretized.fill_(False)
        self.logits.requires_grad_(True)

    def retained_feature_candidates(self, topk: Optional[int] = None):
        """Return retained real-feature indices for every [rule][slot]."""
        probs = self._slot_probabilities().detach()
        out = []
        for r in range(self.out_dim):
            row = []
            for s_idx in range(self.num_slots):
                ids = torch.nonzero(
                    self.candidate_mask[r, s_idx, : self.in_dim], as_tuple=False
                ).squeeze(-1)
                if topk is not None and ids.numel() > int(topk):
                    vals = probs[r, s_idx, ids]
                    ids = ids[torch.topk(vals, int(topk)).indices]
                row.append([int(i) for i in ids.cpu().tolist()])
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # regularisation
    # ------------------------------------------------------------------
    def regularizer(
        self,
        l1_weight: float = 0.0,
        entropy_weight: float = 0.0,
        order_weight: float = 0.0,
    ) -> torch.Tensor:
        """Regularise expected order and categorical confidence.

        ``l1_weight`` penalises non-identity mass of optional slots. Mandatory
        slots are not penalised because they cannot be removed anyway.
        """
        probs = self._slot_probabilities()
        reg = torch.zeros((), device=probs.device, dtype=probs.dtype)

        if l1_weight != 0.0 and self.num_slots > self.min_order:
            optional = probs[:, self.min_order :, :]
            non_identity = 1.0 - optional[..., self.in_dim]
            reg = reg + float(l1_weight) * non_identity.mean()

        if entropy_weight != 0.0:
            p = probs.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=-1).mean()
            reg = reg + float(entropy_weight) * ent

        if order_weight != 0.0 and self.target_order is not None:
            expected_order = (1.0 - probs[..., self.in_dim]).sum(dim=1)
            reg = reg + float(order_weight) * ((expected_order - self.target_order) ** 2).mean()

        # Mild duplicate-mass penalty in the soft surrogate. Hard assignments
        # already forbid duplicates; this keeps the soft model closer to it.
        feat_mass = probs[..., : self.in_dim].sum(dim=1)
        duplicate = torch.relu(feat_mass - 1.0).pow(2).mean()
        reg = reg + 1e-4 * duplicate

        return reg

    # ------------------------------------------------------------------
    # model surgery / explicit masks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_subset(self, in_ids, out_ids):
        in_ids = torch.as_tensor(in_ids, dtype=torch.long, device=self.logits.device)
        out_ids = torch.as_tensor(out_ids, dtype=torch.long, device=self.logits.device)
        new_in = int(in_ids.numel())
        new_out = int(out_ids.numel())
        new_max = min(self.max_order, new_in)
        new_min = min(self.min_order, new_max)

        new = RuleMaskProduct(
            in_dim=new_in,
            out_dim=new_out,
            init_prob=0.5,
            temperature=self.temperature,
            threshold=self.threshold,
            stochastic=self.stochastic,
            min_order=new_min,
            max_order=new_max,
            target_order=self.target_order,
            surrogate_clip=self.surrogate_clip,
            gate_mode=self.gate_mode,
            asinh_scale=float(self.log_asinh_scale.detach().exp().mean()),
            learnable_asinh_scale=self.learnable_asinh_scale,
            rule_orders=(
                None if self.rule_orders is None
                else [min(int(v), new_max) for v in self.rule_orders[out_ids].tolist()]
            ),
        ).to(self.logits.device)

        # Copy matching slots/features and the identity column.
        old = self.logits[out_ids][:, : new.num_slots]
        new.logits[..., :new_in].copy_(old[..., in_ids])
        new.logits[..., new_in].copy_(old[..., self.in_dim])
        new.candidate_mask[..., :new_in].copy_(self.candidate_mask[out_ids][:, : new.num_slots][..., in_ids])
        new.candidate_mask[..., new_in].copy_(self.candidate_mask[out_ids][:, : new.num_slots][..., self.in_dim])
        new.log_asinh_scale.copy_(self.log_asinh_scale[out_ids][:, : new.num_slots])
        new.discretized.copy_(self.discretized)
        new.logits.requires_grad_(self.logits.requires_grad)
        return new

    @torch.no_grad()
    def set_hard_mask(self, mask: torch.Tensor, strength: float = 12.0, freeze: bool = False):
        """Initialise/fix the rule from an explicit binary [out_dim,in_dim] mask."""
        mask = torch.as_tensor(mask, device=self.logits.device)
        if mask.shape != (self.out_dim, self.in_dim):
            raise ValueError(f"mask must have shape {(self.out_dim, self.in_dim)}")

        self.logits.fill_(-float(strength))
        for r in range(self.out_dim):
            ids = torch.nonzero(mask[r] > 0, as_tuple=False).squeeze(-1).tolist()
            if self.rule_orders is not None:
                required = int(self.rule_orders[r].item())
                if len(ids) != required:
                    raise ValueError(
                        f"rule {r} selects {len(ids)} factors but rule_orders requires {required}"
                    )
            else:
                if len(ids) < self.min_order:
                    raise ValueError(
                        f"rule {r} selects {len(ids)} factors but min_order={self.min_order}"
                    )
                if len(ids) > self.max_order:
                    raise ValueError(
                        f"rule {r} selects {len(ids)} factors but max_order={self.max_order}"
                    )

            for s in range(self.num_slots):
                self.candidate_mask[r, s, :] = False
                if s < len(ids):
                    self.logits[r, s, ids[s]] = float(strength)
                    self.candidate_mask[r, s, ids[s]] = True
                else:
                    self.logits[r, s, self.in_dim] = float(strength)
                    self.candidate_mask[r, s, self.in_dim] = True

        self.discretized.fill_(bool(freeze))
        self.logits.requires_grad_(not freeze)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, min_order={self.min_order}, "
            f"max_order={self.max_order}, temperature={self.temperature}, "
            f"stochastic={self.stochastic}, gate_mode={self.gate_mode}, surrogate_clip={self.surrogate_clip}, "
            f"target_order={self.target_order}, rule_orders={None if self.rule_orders is None else self.rule_orders.tolist()}"
        )
