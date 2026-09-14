"""Three builders.

Each compiles one plain-English constraint into a Mealy machine whose states
carry the user's own vocabulary.  When somebody writes "at most 3 refuels",
the memory state they get back is called `refuels_used=2`.
"""

from __future__ import annotations

from aniate.nmdp.machine import Mealy

__all__ = ["Ordering", "Budget", "Deadline"]


def Ordering(sequence, violation_penalty=-1.0, name=None):
    """Visit the listed events in order.

    ``Ordering(["A", "B"])`` is the 3-state DFA for "B must not happen before
    A": `before_A`, `A_visited_B_pending`, and `violated`.  There is no fourth
    state for "both visited", because once A has been seen the constraint can
    never be violated again -- so tracking B would be memory that changes
    nothing, which is exactly what this library exists to avoid.

    For a longer chain the machine has ``len(sequence)`` live stages plus the
    violation sink.
    """
    seq = [str(x) for x in sequence]
    if len(seq) < 2:
        raise ValueError("Ordering needs at least two events")
    if len(set(seq)) != len(seq):
        raise ValueError(f"Ordering events must be distinct; got {seq}")

    k = len(seq)
    stage_names = [f"before_{seq[0]}"] + [
        f"{seq[i - 1]}_visited_{seq[i]}_pending" for i in range(1, k)
    ]
    violated = "violated"
    states = stage_names + [violated]
    rank = {e: i for i, e in enumerate(seq)}

    def transition(q, sigma):
        if q == violated:
            return violated
        stage = stage_names.index(q)
        seen = sorted((rank[e] for e in sigma if e in rank))
        for j in seen:
            if j > stage:
                return violated          # jumped the queue
            if j == stage:
                stage = min(stage + 1, k - 1)
        return stage_names[stage]

    def output(q, sigma):
        if q != violated and transition(q, sigma) == violated:
            return violation_penalty
        return 0.0

    meaning = {stage_names[0]: f"nothing visited yet; {seq[0]} must come first"}
    for i in range(1, k):
        meaning[stage_names[i]] = (
            f"{', '.join(seq[:i])} visited in order; {seq[i]} is now allowed"
        )
    meaning[violated] = "ordering was broken; this is a sink"

    return Mealy(
        name or f"Ordering({' before '.join(seq)})",
        states, stage_names[0], tuple(seq), transition,
        output=output, accepting=stage_names, meaning=meaning,
    )


def Budget(resource, max, violation_penalty=-1.0, label=None, name=None):
    """Use ``resource`` at most ``max`` times.

    ``Budget("refuel", max=3)`` is a 4-state counter: `refuels_used=0` through
    `refuels_used=3`.  The counter saturates rather than growing a fifth
    "exceeded" state -- once you are over budget, *how far* over does not
    change what the optimal policy should do next, and the penalty on the
    over-budget transition already carries the cost.
    """
    if max < 0:
        raise ValueError(f"max must be non-negative; got {max}")
    # `Budget("refuel", ...)` should read `refuels_used=2`, not `refuel_used=2`.
    stem = label or (resource if resource.endswith("s") else resource + "s")
    states = [f"{stem}_used={i}" for i in range(max + 1)]

    def count(q):
        return int(q.rsplit("=", 1)[1])

    def transition(q, sigma):
        if resource in sigma:
            return states[min(count(q) + 1, max)]
        return q

    def output(q, sigma):
        if resource in sigma and count(q) >= max:
            return violation_penalty
        return 0.0

    meaning = {
        states[i]: f"{i} of {max} {stem} used"
        + (" -- any further use is over budget" if i == max else "")
        for i in range(max + 1)
    }
    return Mealy(
        name or f"Budget({resource} <= {max})",
        states, states[0], (resource,), transition,
        output=output, accepting=states, meaning=meaning,
    )


def Deadline(steps, expiry_penalty=-1.0, stop_on=None, name=None):
    """A step counter.

    ``Deadline(steps=50)`` is the 51-state counter `step=0` .. `step=50`.
    For a plain time limit with no other constraint, prefer
    ``MDP(horizon=50)``: backward induction gets the same answer without
    multiplying the state space by 51.  This exists for when a deadline has
    to interact with other memory.

    ``stop_on`` names an event that freezes the clock -- the goal being
    reached, typically -- so that finishing early is distinguishable from
    running out of time.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got {steps}")
    states = [f"step={i}" for i in range(steps + 1)]
    events = (stop_on,) if stop_on else ()

    def transition(q, sigma):
        if stop_on and stop_on in sigma:
            return q
        return states[min(int(q.split("=")[1]) + 1, steps)]

    def output(q, sigma):
        if transition(q, sigma) == states[steps] and q != states[steps]:
            return expiry_penalty
        return 0.0

    meaning = {
        states[0]: "no steps taken yet",
        states[steps]: f"deadline of {steps} steps reached",
    }
    return Mealy(
        name or f"Deadline({steps} steps)",
        states, states[0], events, transition,
        output=output, accepting=states[:steps], meaning=meaning,
    )
