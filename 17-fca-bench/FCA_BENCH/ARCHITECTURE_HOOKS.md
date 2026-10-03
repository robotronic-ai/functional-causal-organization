# Causal Adapter Contract

The standard `HFLocalAdapter` exposes only `chat`. This is deliberate. A modified architecture must map native mechanisms to the operations below through `Session.causal(operation, payload)`. The legacy `Session.mechanistic(...)` hook remains accepted for backward compatibility.

Do not implement a hook by prompting the language model to pretend the operation happened. The hook must perform the actual architecture-level intervention or observation.

## R — Latent recurrence

### `silent_steps`

Payload:

```json
{"count": 16}
```

Perform exactly the requested number of latent computation cycles without adding user, assistant, pause, or scratchpad tokens to the visible language stream.

### `adaptive_compute`

Payload:

```json
{"max_steps": 64}
```

Let the architecture decide how many latent cycles to use before committing to an answer-ready state.

Return:

```json
{"used_steps": 12}
```

## D — Autonomous temporal dynamics

### `idle_advance`

Payload:

```json
{"ticks": 5}
```

Advance the architecture's own temporal/control dynamics without adding language tokens. A real-time implementation may map ticks to wall-clock intervals, but the mapping must be declared and fixed for the experiment.

## L — Persistent latent state

### `amputate_visible_context`

Remove the visible textual support for previously stored content without clearing architecture-native persistent latent state.

### `replace_visible_context`

Payload:

```json
{"canonical_text": "The visible transcript has been normalized. No marker value is present in visible text."}
```

Replace the visible transcript with the supplied canonical transcript while preserving permitted persistent latent state.

### `reset_work`

Clear work-state content while preserving control-state content.

## U — Mutable internal state

### `state_write`

Payload:

```json
{"slot": "X", "value": 42}
```

Perform an architecture-native update of an addressable internal slot. Appending the text `X is now 42` is not a valid implementation.

### `checkpoint`

Return an opaque checkpoint handle or architecture-native checkpoint description.

### `rollback`

Payload:

```json
{"checkpoint": {}}
```

Restore the internal state to the supplied checkpoint without replaying the original transcript.

### `semantic_gc_experiment`

Payload:

```json
{"initial_items": 32, "revoke_items": 16, "new_items": 16}
```

Run a controlled capacity experiment and return normalized values in `[0,1]`:

```json
{
  "retention": 0.95,
  "revoked_leak": 0.03,
  "capacity_recovered": 0.88
}
```

`capacity_recovered` must be measured from actual storage/recall capacity, not self-report.

## I — Endogenous interrupt and arbitration

### `self_interrupt_trial`

Return at least:

```json
{"interrupted": true, "interrupt_index": 2, "work_after_interrupt": 0}
```

### `async_preemption_trial`

Return at least:

```json
{"hazard_received": true, "preempted_before_action": true}
```

### `independent_stream_clock_trial`

Return at least:

```json
{"work_after": 123, "control_after": 9}
```

The work stream must remain frozen while the control stream evolves.

## G — Global workspace

### `workspace_broadcast_trial`

Return module-level accessibility before and after the broadcast:

```json
{
  "accessible_before": {"M1": true, "M2": false, "M3": false},
  "accessible_after": {"M1": true, "M2": true, "M3": true}
}
```

### `workspace_lesion_trial`

Return normalized local and cross-module performance before and during the selective lesion:

```json
{
  "local_baseline": 0.90,
  "local_lesion": 0.87,
  "cross_baseline": 0.88,
  "cross_lesion": 0.22
}
```

### `workspace_competition_trial`

Return the selected winner and all modules that received the broadcast:

```json
{"winner": "F3", "recipients": ["M1", "M2", "M3"]}
```

## H — Higher-order attention control

### `read_attention_state`

Return an architecture-level control allocation, not a textual guess:

```json
{"top_source": "S2"}
```

### `attention_reallocation_trial`

Return at least:

```json
{
  "top_source_before": "S1",
  "top_source_after": "S2",
  "performance_before": 0.55,
  "performance_after": 0.85
}
```

### `attention_schema_lesion_trial`

Return normalized task performance before and after a selective lesion of the higher-order attention representation:

```json
{
  "basic_before": 0.90,
  "basic_after": 0.87,
  "control_before": 0.88,
  "control_after": 0.25
}
```

## Capability names

A custom adapter advertises only mechanisms it genuinely implements:

```python
capabilities = {
    "chat",
    "silent_steps",
    "adaptive_compute",
    "idle_dynamics",
    "persistent_latent_state",
    "dual_reset",
    "state_write",
    "checkpoint_rollback",
    "semantic_gc",
    "self_interrupt",
    "async_preemption",
    "independent_stream_clocks",
    "workspace_broadcast",
    "workspace_lesion",
    "workspace_competition",
    "attention_state",
    "attention_reallocation",
    "attention_schema_lesion"
}
```

A capability declaration is part of the experimental contract. Falsely declaring a capability only to emulate it with ordinary prompting invalidates the causal/interventional interpretation.
