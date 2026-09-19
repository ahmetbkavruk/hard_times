"""A transparent, stochastic local controller; no alignment-specific scripts."""

from math import ceil, exp, log

import numpy as np

from .actions import Action, ActionKind
from .policies import Observation, RuleBasedPolicy, TradeOffer


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def distance_after(dx: int, dy: int, direction: tuple[int, int], length: int) -> int:
    """Periodic Manhattan distance after a proposed one-cell move."""
    x, y = (dx - direction[0]) % length, (dy - direction[1]) % length
    return min(x, length - x) + min(y, length - y)


class SocialPolicy(RuleBasedPolicy):
    """Score available actions using needs, effort, local opportunities and memory.

    Scores are modeling assumptions, not learned intelligence or evidence for
    any emergent outcome. Alignment names/colors never enter this controller.
    """

    def _needs(self, observation):
        stocks = np.concatenate((observation.inventory[:1], np.asarray(
            observation.self_produced_inventory
        ) + np.asarray(observation.received_inventory)))
        recipe = np.asarray(self.config.recipe)
        return stocks, np.maximum(0, recipe - stocks) / np.maximum(recipe, 1e-12)

    def action_distribution(self, observation: Observation):
        cfg = self.config
        stocks, needs = self._needs(observation)
        hunger = max(0.0, (cfg.hungry_energy - observation.energy) / max(cfg.hungry_energy, 1e-12))
        care = (observation.concern + 1) / 2
        consistency = (observation.lawfulness + 1) / 2
        target_energy = cfg.reproduce_threshold + cfg.child_energy + cfg.body_energy
        candidates = []

        def add(action, utility):
            candidates.append((action, float(utility)))

        if np.all(stocks >= cfg.recipe) and observation.energy < target_energy:
            add(Action(ActionKind.USE), 6 + 2 * hunger)

        if observation.local_resources[0] > 0 and stocks[0] < 2 * cfg.recipe[0]:
            gain = min(observation.local_resources[0], cfg.q_max * observation.genotype[0])
            if gain > 0:
                add(Action(ActionKind.HARVEST, resource=0),
                    1 + 3 * min(needs[0], gain / max(cfg.recipe[0], 1e-12)) - cfg.harvest_cost)

        allocation = observation.production_allocation
        output = cfg.production_amount * np.array([
            allocation ** cfg.production_specialization_exponent,
            (1 - allocation) ** cfg.production_specialization_exponent,
        ])
        useful_output = np.minimum(needs[1:], output / np.maximum(cfg.recipe[1:], 1e-12)).sum()
        # Producing a small export reserve can enable later exchange. Unbounded
        # stockpiling has no utility, and a missing zero-output good favors search.
        reserve_needed = cfg.trade_enabled and any(
            amount > 0 and own < recipe + cfg.max_trade_per_pair
            for amount, own, recipe in zip(output, observation.self_produced_inventory, cfg.recipe[1:])
        )
        if useful_output > 0 or (reserve_needed and np.any(needs[1:] > 0)):
            add(Action(ActionKind.PRODUCE), 1 + 3 * useful_output - cfg.production_cost)

        if (observation.energy >= max(cfg.reproduce_threshold, cfg.reproduce_cost + cfg.child_energy + cfg.body_energy)
                and observation.population < cfg.max_agents and cfg.reproduce_probability > 0):
            add(Action(ActionKind.REPRODUCE),
                4 + min(2, (observation.energy - cfg.reproduce_threshold) / 5)
                + log(cfg.reproduce_probability / 0.12))

        threats = [neighbor for neighbor in observation.neighbors if neighbor.harmed_me]
        for direction in DIRECTIONS:
            value = -cfg.move_cost_scale * (1 - observation.genotype[1])
            resource = next((amount for dx, dy, amount in observation.adjacent_resources
                             if (dx, dy) == direction), observation.local_resources[0])
            value += 2.5 * needs[0] * np.clip(
                (resource - observation.local_resources[0]) / max(cfg.q_max, 1e-12), -1, 1
            )
            if observation.last_direction == direction:
                value += 0.4 * consistency
            for neighbor in observation.neighbors:
                before = abs(neighbor.dx) + abs(neighbor.dy)
                after = distance_after(neighbor.dx, neighbor.dy, direction, observation.grid_size)
                approach = before - after
                if neighbor.harmed_me:
                    value -= 3 * approach
                elif cfg.trade_enabled and before > 1:
                    # Seek a contact only when both sides could fill a deficit.
                    complementary = any(
                        needs[receive_good] > 0
                        and neighbor.self_produced_inventory[receive_good - 1] > cfg.recipe[receive_good]
                        and observation.self_produced_inventory[give_good - 1] > cfg.recipe[give_good]
                        and neighbor.stock(give_good) < cfg.recipe[give_good]
                        for give_good, receive_good in ((1, 2), (2, 1))
                    )
                    if complementary:
                        value += 2 * approach * (1 + 0.2 * consistency * min(neighbor.recent_trades, 3))
            if observation.energy < target_energy:
                for body in observation.bodies:
                    before = abs(body.dx) + abs(body.dy)
                    if before > 1:
                        value += (1 + hunger) * (before - distance_after(
                            body.dx, body.dy, direction, observation.grid_size
                        ))
            add(Action(ActionKind.MOVE, direction=direction), value)
            if threats and min(distance_after(n.dx, n.dy, direction, observation.grid_size) for n in threats) > min(
                abs(n.dx) + abs(n.dy) for n in threats
            ):
                add(Action(ActionKind.FLEE, direction=direction), value + 1 - cfg.flee_cost)

        damage = cfg.attack_damage * (0.5 + observation.genotype[0])
        if damage > 0 and observation.energy > cfg.attack_cost:
            for neighbor in observation.neighbors:
                if abs(neighbor.dx) + abs(neighbor.dy) > 1:
                    continue
                if not neighbor.harmed_me and (hunger <= 0 or neighbor.body_energy <= 0):
                    continue
                hunting_cost = ceil(neighbor.energy / damage) * (cfg.attack_cost + cfg.metabolism)
                benefit = cfg.predation_efficiency * neighbor.body_energy - hunting_cost
                risk = 2 * max(0, neighbor.energy / max(observation.energy, 1e-12) - 0.5)
                value = -1 + hunger * benefit - 2.5 * care - risk
                if neighbor.harmed_me:
                    value += 4  # Defensive value depends on an actual remembered attack.
                add(Action(ActionKind.ATTACK, target_id=neighbor.id), value)

        if observation.energy < target_energy and observation.energy > cfg.eating_cost:
            for body in observation.bodies:
                if abs(body.dx) + abs(body.dy) <= 1 and body.edible_energy > 0 and cfg.bite_energy > 0:
                    net = min(body.edible_energy, cfg.bite_energy) * cfg.predation_efficiency
                    add(Action(ActionKind.EAT_AGENT, target_id=body.agent_id),
                        2 + (1 + hunger) * net - cfg.eating_cost)

        utilities = np.array([value for _, value in candidates])
        # Chaotic agents explore more; lawful agents make more consistent choices.
        temperature = cfg.decision_temperature * (1.25 - 0.5 * observation.lawfulness)
        weights = np.exp(np.clip((utilities - utilities.max()) / temperature, -700, 0))
        probabilities = weights / weights.sum()
        return [(action, float(probability)) for (action, _), probability in zip(candidates, probabilities)]

    def choose_action(self, observation, rng):
        distribution = self.action_distribution(observation)
        index = int(rng.choice(len(distribution), p=[probability for _, probability in distribution]))
        return distribution[index][0]

    def trade_probability(self, observation: Observation, offer: TradeOffer) -> float:
        """Evaluate this actual offer; no commitment or alliance is inferred."""
        cfg = self.config
        if observation.energy <= cfg.trade_energy_cost or offer.net <= 0:
            return 0.0
        own_benefit = offer.net / max(cfg.recipe[offer.receive_good], 1e-12)
        partner_benefit = offer.net / max(cfg.recipe[offer.give_good], 1e-12)
        consistency = (observation.lawfulness + 1) / 2
        trust = 0.6 * consistency * min(offer.partner.recent_trades, 3)
        danger = (2 + 2 * consistency) if offer.partner.harmed_me else 0
        value = 2 * own_benefit + observation.concern * partner_benefit + trust - danger - cfg.trade_energy_cost
        return 1 / (1 + exp(-max(-700, min(700, value))))

    def willing_to_trade(self, observation, offer, rng) -> bool:
        return bool(rng.random() < self.trade_probability(observation, offer))
