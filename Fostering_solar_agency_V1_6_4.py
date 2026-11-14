import random
import math
import time
import os
import csv
from datetime import datetime


def sigmoid(x):
    x = max(-20, min(20, x))          # clamp to avoid overflow
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(s):
    """Derivative of sigmoid given its *output* s (not input)."""
    return s * (1 - s)


# ═══════════════════════════════════════════════════════════════════════════════
# TinyMLP  –  a tiny 2-layer MLP trained online via policy-gradient-style update
#
# Architecture:  6 inputs → 4 hidden (sigmoid) → 1 output (sigmoid)
#
# Inputs (all normalised to ≈[0,1]):
#   0  stored_energy / storage_capacity      – how full the battery is
#   1  production / solar_capacity           – current solar output fraction
#   2  sin(t)                                – time-of-day proxy (peak = 1)
#   3  budget / 5000                         – financial headroom
#   4  ev_charge / (ev_capacity + ε)         – EV state of charge
#   5  base_consumption / 10.0              – household demand proxy
#
# Output:  decision ∈ (0,1)
#   • close to 1 → use more energy now (solar is abundant)
#   • close to 0 → conserve (solar is scarce, battery low)
#
# Training signal (one scalar reward per tick):
#   reward = self_consumption_rate − overflow_penalty − deficit_penalty
#
#   self_consumption_rate  = solar_used_locally / (production + ε)
#   overflow_penalty       = overflow / (storage_capacity + ε)   [wasted solar]
#   deficit_penalty        = |deficit| / (storage_capacity + ε)  [unmet need]
#
# Update rule (REINFORCE-lite, simplified):
#   δ = η × (reward − baseline) × sigmoid_deriv(output) × hidden_activation
#
#  CSV routine added
#
# ═══════════════════════════════════════════════════════════════════════════════

class TinyMLP:
    """
    Tiny 2-layer MLP (6→4→1) trained online to maximise solar self-consumption.
    Weights are updated after every simulation tick using a lightweight
    policy-gradient-style rule – no external library required.
    """

    N_IN  = 6
    N_HID = 4
    N_OUT = 1

    def __init__(self, lr=0.05, baseline_decay=0.95):
        # Xavier-ish init: scale by 1/sqrt(fan_in)
        scale_ih = 1 / math.sqrt(self.N_IN)
        scale_ho = 1 / math.sqrt(self.N_HID)

        # Hidden layer: N_HID × N_IN weights + N_HID biases
        self.W1 = [[random.uniform(-scale_ih, scale_ih)
                    for _ in range(self.N_IN)]
                   for _ in range(self.N_HID)]
        self.b1 = [0.0] * self.N_HID

        # Output layer: N_OUT × N_HID weights + N_OUT biases
        self.W2 = [[random.uniform(-scale_ho, scale_ho)
                    for _ in range(self.N_HID)]
                   for _ in range(self.N_OUT)]
        self.b2 = [0.0] * self.N_OUT

        self.lr = lr
        self.baseline      = 0.0        # exponential moving average of reward
        self.baseline_decay = baseline_decay

        # Cached activations for use in train()
        self._last_inputs  = None
        self._last_hidden  = None
        self._last_output  = None

        # Diagnostics
        self.reward_history = []
        self.ticks_trained  = 0

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, inputs: list[float]) -> float:
        """Run the network and cache activations.  Returns a scalar in (0,1)."""
        self._last_inputs = inputs[:]

        # Hidden layer
        hidden = []
        for j in range(self.N_HID):
            z = self.b1[j] + sum(self.W1[j][i] * inputs[i] for i in range(self.N_IN))
            hidden.append(sigmoid(z))
        self._last_hidden = hidden

        # Output layer
        out_z = self.b2[0] + sum(self.W2[0][j] * hidden[j] for j in range(self.N_HID))
        output = sigmoid(out_z)
        self._last_output = output

        return output

    # ── Backward pass (online REINFORCE-lite) ─────────────────────────────────
    def train(self, reward: float):
        """Update weights given the scalar reward for the last forward pass."""
        if self._last_inputs is None:
            return                      # nothing cached yet

        self.ticks_trained += 1
        self.reward_history.append(reward)

        # Update exponential baseline
        self.baseline = (self.baseline_decay * self.baseline
                         + (1 - self.baseline_decay) * reward)
        advantage = reward - self.baseline

        # --- Output layer gradient ---
        # dL/d(out_z) = advantage × σ'(output)
        delta_out = advantage * sigmoid_deriv(self._last_output)

        # Update W2 and b2
        for j in range(self.N_HID):
            self.W2[0][j] += self.lr * delta_out * self._last_hidden[j]
        self.b2[0] += self.lr * delta_out

        # --- Hidden layer gradient ---
        for j in range(self.N_HID):
            # Back-prop through output weight
            delta_hid = delta_out * self.W2[0][j] * sigmoid_deriv(self._last_hidden[j])
            for i in range(self.N_IN):
                self.W1[j][i] += self.lr * delta_hid * self._last_inputs[i]
            self.b1[j] += self.lr * delta_hid

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def avg_reward(self, n=50) -> float:
        """Mean reward over the last n ticks."""
        if not self.reward_history:
            return 0.0
        window = self.reward_history[-n:]
        return sum(window) / len(window)


# ═══════════════════════════════════════════════════════════════════════════════
class Household:

    ENERGY_CENTERED = "energy"
    CONSUMPTION_CENTERED = "consumption"

    ARCHETYPES = {
        1: {"name": "Solar Neglect",    "symbol": "N"},
        2: {"name": "Solar Disengaged", "symbol": "D"},
        3: {"name": "Solar Constrained","symbol": "C"},
        4: {"name": "Solar Pragmatic",  "symbol": "P"},
        5: {"name": "Solar Critical",   "symbol": "R"},
        6: {"name": "No/Low Solar",     "symbol": "L"},
    }

    # Per-tick grid access cost, offset by solar production.
    # Drains budgets over time so poor producers eventually collapse.
    BASELINE_COST = {
        1: 4.0,   # Neglect: moderate drain
        2: 3.0,   # Disengaged
        3: 2.5,   # Constrained
        4: 0.8,   # Pragmatic: nearly self-sufficient
        5: 2.0,   # Critical
        6: 6.0,   # No/Low Solar: highest drain
    }

    # Rebirth constants
    REBIRTH_THRESHOLD = 50.0   # minimum pooled budget needed to revive a cell
    DONATION_BUDGET   = 25.0   # budget each willing neighbour donates
    DONATION_ENERGY   = 2.0    # energy each willing neighbour donates

    def __init__(self, x, y, state_type):
        self.x = x
        self.y = y
        self.state_type = state_type

        self.name        = self.ARCHETYPES[state_type]["name"]
        self.symbol_code = self.ARCHETYPES[state_type]["symbol"]

        self._init_stats()

    def _init_stats(self):
        """Initialise (or re-initialise on rebirth) all household stats."""
        state_type = self.state_type

        # ---------- Population ----------
        pop_ranges = {1:(2,4),2:(2,5),3:(3,6),4:(2,4),5:(2,5),6:(3,7)}
        self.population = random.randint(*pop_ranges[state_type])

        # ---------- Income ----------
        income_ranges = {
            1:(1100,1800),2:(500,1100),3:(500,1100),
            4:(1100,1800),5:(500,1100),6:(150,500),
        }
        self.income = random.uniform(*income_ranges[state_type])

        # Tighter starting budgets so collapse can actually occur
        budget_ranges = {
            1:(1.5,2.5), 2:(1.5,2.5), 3:(1.5,2.5),
            4:(2.0,3.5),               # Pragmatic gets a bit more runway
            5:(1.5,2.5), 6:(1.0,1.8), # No/Low Solar starts thinnest
        }
        self.budget = self.income * random.uniform(*budget_ranges[state_type])

        # ---------- Solar ----------
        solar_ranges = {
            1:(0,2),2:(1,3),3:(0.5,2),4:(7,12),5:(2,5),6:(0,1),
        }
        self.solar_capacity = random.uniform(*solar_ranges[state_type])

        # ---------- Battery ----------
        storage_ranges = {
            1:(10,18),2:(8,14),3:(5,10),4:(18,30),5:(10,16),6:(3,6),
        }
        self.storage_capacity = random.uniform(*storage_ranges[state_type])
        self.stored_energy    = random.uniform(0, self.storage_capacity * 0.3)

        # ---------- Efficiency ----------
        efficiency_ranges = {
            1:(0.7,0.9),2:(0.8,1.0),3:(0.9,1.1),
            4:(0.6,0.8),5:(0.8,1.0),6:(1.0,1.3),
        }
        self.appliance_efficiency = random.uniform(*efficiency_ranges[state_type])

        # ---------- Behaviour ----------
        self.behaviour = (
            self.ENERGY_CENTERED
            if state_type in [1, 4]
            else self.CONSUMPTION_CENTERED
        )

        # ---------- EV ----------
        self.has_ev = random.random() < {
            1:0.7,2:0.4,3:0.2,4:0.7,5:0.4,6:0.1
        }[state_type]

        if self.has_ev:
            self.ev_capacity = random.uniform(20, 60)
            self.ev_charge   = random.uniform(0.2, 0.8) * self.ev_capacity
        else:
            self.ev_capacity = 0
            self.ev_charge   = 0

        # ──────────────────────────────────────────────────────────────────────
        # TinyMLP — fresh brain on every init (including rebirth)
        # ──────────────────────────────────────────────────────────────────────
        self.brain = TinyMLP(lr=0.05)

        # metrics
        self.wastefulness = 0
        self.comfort      = 0
        self.alive        = True
        self.ticks_dead   = 0

        # cost/benefit accumulators (reset each tick)
        self.cost_deficit   = 0.0
        self.cost_bought    = 0.0
        self.benefit_export = 0.0
        self.benefit_sold   = 0.0

        # track last-tick values so we can compute the reward
        self._last_production       = 0.0
        self._last_overflow         = 0.0
        self._last_deficit          = 0.0
        self._last_solar_used_local = 0.0

    def revive(self, donated_budget, donated_energy):
        """Called by Grid.rebirth() when neighbours pool enough resources."""
        self._init_stats()
        self.budget        = donated_budget
        self.stored_energy = min(donated_energy, self.storage_capacity)
        self.alive         = True
        self.ticks_dead    = 0

    # ── Helpers ────────────────────────────────────────────────────────────────
    def solar_production(self, t):
        return max(0, math.sin(t)) * self.solar_capacity

    def base_consumption(self):
        return self.population * 1.2 * self.appliance_efficiency

    # ── Feature extraction ────────────────────────────────────────────────────
    def _build_inputs(self, production: float, t: float) -> list[float]:
        """Return the 6-element normalised input vector for the TinyMLP."""
        return [
            self.stored_energy / (self.storage_capacity + 1e-6),   # 0: SOC
            production / (self.solar_capacity + 1e-6),              # 1: solar fraction
            (math.sin(t) + 1) / 2,                                  # 2: time-of-day [0,1]
            min(self.budget / 5000.0, 1.0),                         # 3: budget headroom
            self.ev_charge / (self.ev_capacity + 1e-6),             # 4: EV SOC
            min(self.base_consumption() / 10.0, 1.0),               # 5: demand proxy
        ]

    # ── Reward computation ────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        """
        Scalar reward ∈ [-1, 1] that rewards solar self-consumption and
        penalises wasted overflow and deficit.

        Components:
          + self_consumption_rate  (solar actually used locally / produced)
          - overflow_rate          (solar wasted to grid due to full battery)
          - deficit_rate           (energy need unmet)
        """
        prod  = self._last_production + 1e-6
        cap   = self.storage_capacity + 1e-6

        self_consumption_rate = min(self._last_solar_used_local / prod, 1.0)
        overflow_rate         = min(self._last_overflow / cap, 1.0)
        deficit_rate          = min(self._last_deficit  / cap, 1.0)

        reward = self_consumption_rate - 0.6 * overflow_rate - 0.8 * deficit_rate
        return max(-1.0, min(1.0, reward))

    # ── EV charging ────────────────────────────────────────────────────────────
    def ev_charging(self, available_energy, decision_factor):
        if not self.has_ev:
            return 0

        desired = self.ev_capacity * 0.8
        if self.ev_charge >= desired:
            return 0

        demand = desired - self.ev_charge

        if self.behaviour == self.ENERGY_CENTERED:
            aggressiveness = 0.6 + 0.4 * decision_factor
        else:
            aggressiveness = 0.8

        requested  = min(demand, random.uniform(1.0, 3.0) * aggressiveness)
        safe_limit = available_energy * 0.7
        actual     = min(requested, safe_limit)

        self.ev_charge += actual
        return actual

    # ── Main step ──────────────────────────────────────────────────────────────
    def step(self, t):
        if not self.alive:
            self.ticks_dead += 1
            return

        # Reset per-tick accumulators
        self.cost_deficit   = 0.0
        self.cost_bought    = 0.0
        self.benefit_export = 0.0
        self.benefit_sold   = 0.0

        production = self.solar_production(t)
        base       = self.base_consumption()

        # ── TinyMLP forward pass ──────────────────────────────────────────────
        inputs   = self._build_inputs(production, t)
        decision = self.brain.forward(inputs)          # ← replaces old heuristic
        # decision ∈ (0,1):  high → use more solar now,  low → conserve

        # ── EV charging ───────────────────────────────────────────────────────
        energy_buffer = self.storage_capacity * 0.15
        available     = max(0, production + self.stored_energy - energy_buffer)
        ev_use        = self.ev_charging(available, decision)

        # ── Main consumption ──────────────────────────────────────────────────
        if self.behaviour == self.ENERGY_CENTERED:
            consumption = base * (0.8 + 0.6 * decision)
        else:
            consumption = base * 1.5

        effective_consumption = consumption / (self.appliance_efficiency + 1e-6)

        # ── Energy balance ────────────────────────────────────────────────────
        # Track how much solar was consumed locally (before battery clipping)
        solar_into_store   = max(0, production - effective_consumption - ev_use)
        solar_used_locally = production - max(0, solar_into_store - (self.storage_capacity - self.stored_energy))
        solar_used_locally = max(0, min(solar_used_locally, production))

        self.stored_energy += production - effective_consumption - ev_use

        # ── V2H discharge ─────────────────────────────────────────────────────
        v2h_discharge = 0.0
        if self.stored_energy < self.storage_capacity * 0.2:
            urgency       = 1 - (self.stored_energy / (self.storage_capacity * 0.2))
            v2h_discharge = min(self.ev_charge, 2.0 * urgency * decision)
            self.ev_charge     -= v2h_discharge
            self.stored_energy += v2h_discharge

        # ── Overflow / deficit ────────────────────────────────────────────────
        overflow = 0.0
        if self.stored_energy > self.storage_capacity:
            overflow            = self.stored_energy - self.storage_capacity
            earned              = overflow * 0.4
            self.budget        += earned
            self.benefit_export = earned
            self.stored_energy  = self.storage_capacity

        deficit = 0.0
        if self.stored_energy < 0:
            deficit             = min(-self.stored_energy, self.storage_capacity * 0.5)
            self.stored_energy  = 0
            # Archetype-specific penalty: Pragmatic is buffered, No/Low Solar pays most
            penalty_rate = {1:1.2, 2:1.0, 3:0.9, 4:0.7, 5:1.0, 6:1.8}[self.state_type]
            penalty             = deficit * penalty_rate
            self.cost_deficit  += penalty
            self.budget        -= penalty

        # ── Baseline grid cost (offset by solar production) ───────────────────
        # Recurring drain so poor producers bleed budget every tick
        baseline_cost = self.BASELINE_COST[self.state_type]
        solar_offset  = min(baseline_cost, production * 0.5)
        net_baseline  = baseline_cost - solar_offset
        self.budget       -= net_baseline
        self.cost_deficit += net_baseline

        # ── Comfort & wastefulness ────────────────────────────────────────────
        ideal        = base
        unmet        = max(0, ideal - consumption)
        self.comfort = max(0, min(1, 1 - unmet / (ideal + 1e-6)))
        if self.stored_energy < self.storage_capacity * 0.2:
            self.comfort *= 0.9

        self.wastefulness = sigmoid(overflow / (self.storage_capacity + 1e-6))

        # ── Cache tick outcomes for reward ────────────────────────────────────
        self._last_production       = production
        self._last_overflow         = overflow
        self._last_deficit          = deficit
        self._last_solar_used_local = solar_used_locally

        # ── TinyMLP backward pass (online training) ───────────────────────────
        reward = self._compute_reward()
        self.brain.train(reward)

        # ── Collapse check ────────────────────────────────────────────────────
        if self.budget < 0:
            self.alive = False

    def symbol(self):
        return "." if not self.alive else self.symbol_code

# ═══════════════════════════════════════════════════════════════════════════════
class Grid:

    def __init__(self, w, h):
        self.w    = w
        self.h    = h
        self.time = 0
        self.cells = [
            [Household(x, y, random.randint(1, 6)) for x in range(w)]
            for y in range(h)
        ]
        self.market_price   = 0.3
        self.total_rebirths = 0
        self.tick   = 0
        self.logger = CSVLogger()

    def step(self):
        self.time += 0.2

        for row in self.cells:
            for cell in row:
                cell.step(self.time)

        total_energy   = sum(c.stored_energy    for row in self.cells for c in row)
        total_capacity = sum(c.storage_capacity for row in self.cells for c in row)

        scarcity          = 1 - (total_energy / (total_capacity + 1e-6))
        self.market_price = 0.2 + 0.8 * sigmoid(5 * (scarcity - 0.5))

        self.exchange()
        self.rebirth()
        self.tick += 1
        self.logger.log(self)

    def neighbours(self, x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                yield self.cells[ny][nx]

    def rebirth(self):
        """
        For each collapsed cell, willing neighbours pool budget and energy.
        A neighbour is willing if they have >15% stored energy and budget
        above 80% of their income. If the pool meets the threshold the
        cell revives with the donated resources — genuinely fragile.
        """
        for y in range(self.h):
            for x in range(self.w):
                cell = self.cells[y][x]
                if cell.alive:
                    continue

                donors = [
                    n for n in self.neighbours(x, y)
                    if n.alive
                    and n.stored_energy > n.storage_capacity * 0.15
                    and n.budget > n.income * 0.8
                ]

                if not donors:
                    continue

                pooled_budget = len(donors) * Household.DONATION_BUDGET
                pooled_energy = len(donors) * Household.DONATION_ENERGY

                if pooled_budget < Household.REBIRTH_THRESHOLD:
                    continue    # not enough neighbours — stays dead

                for n in donors:
                    n.budget        -= Household.DONATION_BUDGET
                    n.stored_energy -= Household.DONATION_ENERGY

                cell.revive(pooled_budget, pooled_energy)
                self.total_rebirths += 1

    def exchange(self):
        indices = [(x, y) for y in range(self.h) for x in range(self.w)]
        random.shuffle(indices)

        for x, y in indices:
            cell = self.cells[y][x]
            if not cell.alive:
                continue

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    n = self.cells[ny][nx]
                    if not n.alive:
                        continue

                    if (cell.stored_energy < cell.storage_capacity * 0.3 and
                            n.stored_energy > n.storage_capacity * 0.5):

                        sender_surplus   = n.stored_energy - n.storage_capacity * 0.5
                        sender_offer     = sender_surplus * 0.10
                        receiver_gap     = cell.storage_capacity * 0.3 - cell.stored_energy
                        transfer         = max(0, min(sender_offer, receiver_gap, n.stored_energy))

                        max_affordable   = cell.budget * 0.05
                        max_transferable = max_affordable / (self.market_price + 1e-6)
                        transfer         = max(0, min(transfer, max_transferable))

                        if transfer <= 0:
                            continue

                        payment = transfer * self.market_price

                        n.stored_energy    -= transfer
                        cell.stored_energy += transfer
                        cell.budget        -= payment
                        cell.cost_bought   += payment
                        n.budget           += payment
                        n.benefit_sold     += payment

    def display(self):
        os.system("cls" if os.name == "nt" else "clear")
        print("Grid:  (. = collapsed, awaiting rescue)")
        for row in self.cells:
            print(" ".join(c.symbol() for c in row))

    def stats(self):
        all_cells = [c for row in self.cells for c in row]
        alive     = [c for c in all_cells if c.alive]

        if not alive:
            print("\nAll households collapsed.")
            return

        # Fixed-width column spec — every value has an explicit width so
        # rows stay aligned regardless of sign or magnitude.
        # Columns:  Archetype(22) N(3) Comfort(7) Waste(7)
        #           Cost:Deficit(12) Cost:Bought(11)
        #           Ben:Export(10) Ben:Sold(10) Net(10) ML-Rew(9)
        COL = "{:<22} {:>3}  {:>7} {:>7}  {:>12}  {:>11}  {:>10}  {:>10}  {:>10}  {:>9}"
        SEP = "─" * 112

        print("\n── Per-archetype averages (alive households) " + "─" * 68)
        print(COL.format(
            "Archetype", "N",
            "Comfort", "Waste",
            "Cost:Deficit", "Cost:Bought",
            "Ben:Export", "Ben:Sold", "Net", "ML-Reward"
        ))
        print(SEP)

        for archetype_id, info in Household.ARCHETYPES.items():
            group = [c for c in alive if c.state_type == archetype_id]
            label = f"[{info['symbol']}] {info['name']}"
            if not group:
                print(COL.format(label, "–", "–", "–", "–", "–", "–", "–", "–", "–"))
                continue

            n        = len(group)
            comfort  = sum(c.comfort        for c in group) / n
            waste    = sum(c.wastefulness   for c in group) / n
            deficit  = sum(c.cost_deficit   for c in group) / n
            bought   = sum(c.cost_bought    for c in group) / n
            export_b = sum(c.benefit_export for c in group) / n
            sold     = sum(c.benefit_sold   for c in group) / n
            net      = export_b + sold - deficit - bought 
            ml_rew   = sum(c.brain.avg_reward() for c in group) / n

            print(COL.format(
                label, n,
                f"{comfort:>7.3f}", f"{waste:>7.3f}",
                f"{deficit:>12.3f}", f"{bought:>11.3f}",
                f"{export_b:>10.3f}", f"{sold:>10.3f}",
                f"{net:>+10.3f}", f"{ml_rew:>+9.3f}"
            ))

        # ── Grid averages ─────────────────────────────────────────────────────
        n_alive      = len(alive)
        avg_comfort  = sum(c.comfort        for c in alive) / n_alive
        avg_waste    = sum(c.wastefulness   for c in alive) / n_alive
        avg_deficit  = sum(c.cost_deficit   for c in alive) / n_alive
        avg_bought   = sum(c.cost_bought    for c in alive) / n_alive
        avg_export   = sum(c.benefit_export for c in alive) / n_alive
        avg_sold     = sum(c.benefit_sold   for c in alive) / n_alive
        grid_net     = avg_export + avg_sold - avg_deficit - avg_bought
        avg_rew      = sum(c.brain.avg_reward() for c in alive) / n_alive

        print(SEP)
        print(COL.format(
            f"GRID AVG  ({n_alive}/{self.w * self.h} alive, {self.total_rebirths} rebirths)", "",
            f"{avg_comfort:>7.3f}", f"{avg_waste:>7.3f}",
            f"{avg_deficit:>12.3f}", f"{avg_bought:>11.3f}",
            f"{avg_export:>10.3f}", f"{avg_sold:>10.3f}",
            f"{grid_net:>+10.3f}", f"{avg_rew:>+9.3f}"
        ))
        print(SEP)

# ═══════════════════════════════════════════════════════════════════════════════
class CSVLogger:
    HOUSEHOLD_FIELDS = [
        "tick", "sim_time", "x", "y", "archetype", "alive",
        "population", "income", "budget",
        "solar_capacity", "storage_capacity", "stored_energy",
        "has_ev", "ev_charge", "comfort", "wastefulness",
        "cost_deficit", "cost_bought", "benefit_export", "benefit_sold",
        "net", "ml_avg_reward", "ml_ticks_trained", "market_price",
    ]
    SUMMARY_FIELDS = [
        "tick", "sim_time", "group", "n_alive", "comfort", "wastefulness",
        "cost_deficit", "cost_bought", "benefit_export", "benefit_sold",
        "net", "ml_avg_reward", "total_rebirths", "market_price",
    ]

    def __init__(self, prefix="sim"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(folder, exist_ok=True)
        self.household_path = os.path.join(folder, f"{prefix}_households_{stamp}.csv")
        self.summary_path   = os.path.join(folder, f"{prefix}_summary_{stamp}.csv")

        self._hf = open(self.household_path, "w", newline="")
        self._sf = open(self.summary_path,   "w", newline="")
        self._hw = csv.DictWriter(self._hf, fieldnames=self.HOUSEHOLD_FIELDS)
        self._sw = csv.DictWriter(self._sf, fieldnames=self.SUMMARY_FIELDS)
        self._hw.writeheader()
        self._sw.writeheader()

    @staticmethod
    def _avg(group, attr):
        return sum(attr(c) for c in group) / len(group)

    def log(self, grid):
        tick = grid.tick
        t    = round(grid.time, 4)
        mp   = round(grid.market_price, 5)

        # ── Per-household rows ────────────────────────────────────────────
        for row in grid.cells:
            for c in row:
                net = (c.benefit_export + c.benefit_sold
                       - c.cost_deficit - c.cost_bought)
                self._hw.writerow({
                    "tick": tick, "sim_time": t, "x": c.x, "y": c.y,
                    "archetype": c.name, "alive": int(c.alive),
                    "population": c.population,
                    "income": round(c.income, 3),
                    "budget": round(c.budget, 3),
                    "solar_capacity": round(c.solar_capacity, 3),
                    "storage_capacity": round(c.storage_capacity, 3),
                    "stored_energy": round(c.stored_energy, 3),
                    "has_ev": int(c.has_ev),
                    "ev_charge": round(c.ev_charge, 3),
                    "comfort": round(c.comfort, 4),
                    "wastefulness": round(c.wastefulness, 4),
                    "cost_deficit": round(c.cost_deficit, 4),
                    "cost_bought": round(c.cost_bought, 4),
                    "benefit_export": round(c.benefit_export, 4),
                    "benefit_sold": round(c.benefit_sold, 4),
                    "net": round(net, 4),
                    "ml_avg_reward": round(c.brain.avg_reward(), 4),
                    "ml_ticks_trained": c.brain.ticks_trained,
                    "market_price": mp,
                })

        # ── Summary rows (alive households only, like stats()) ────────────
        all_cells = [c for row in grid.cells for c in row]
        alive     = [c for c in all_cells if c.alive]

        groups = [(info["name"], [c for c in alive if c.state_type == aid])
                  for aid, info in Household.ARCHETYPES.items()]
        groups.append(("GRID", alive))

        for name, grp in groups:
            row = {"tick": tick, "sim_time": t, "group": name,
                   "n_alive": len(grp),
                   "total_rebirths": grid.total_rebirths,
                   "market_price": mp}
            if grp:
                a = self._avg
                exp = a(grp, lambda c: c.benefit_export)
                sold = a(grp, lambda c: c.benefit_sold)
                dfc = a(grp, lambda c: c.cost_deficit)
                bgt = a(grp, lambda c: c.cost_bought)
                row.update({
                    "comfort": round(a(grp, lambda c: c.comfort), 4),
                    "wastefulness": round(a(grp, lambda c: c.wastefulness), 4),
                    "cost_deficit": round(dfc, 4),
                    "cost_bought": round(bgt, 4),
                    "benefit_export": round(exp, 4),
                    "benefit_sold": round(sold, 4),
                    "net": round(exp + sold - dfc - bgt, 4),
                    "ml_avg_reward": round(a(grp, lambda c: c.brain.avg_reward()), 4),
                })
            self._sw.writerow(row)

        # Flush so data survives Ctrl+C
        self._hf.flush()
        self._sf.flush()

    def close(self):
        self._hf.close()
        self._sf.close()

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    g = Grid(15, 8)
    try:
        while True:
            g.display()
            g.step()
            g.stats()
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        g.logger.close()
        print(f"\nSaved:\n  {g.logger.household_path}\n  {g.logger.summary_path}")