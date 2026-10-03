# Fresh and Sealed Evaluation Instances

## Goal

The sealing workflow makes instance selection auditable and reduces exact-item training contamination and post-hoc item selection.

It does not make a public generator family secret and does not prove that no structurally similar training example existed.

## Create

Generate a bank only after the evaluated model and benchmark version are frozen:

```bash
python -m fca_bench.sealing create --suite all --n 20 \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

The private file contains the random seeds used by deterministic FCA instance generators.

The public file contains:

- bank identifier;
- protocol version;
- number of instances per test;
- SHA-256 commitment for every test seed;
- SHA-256 commitment for the complete private bank.

## Commit

Before executing the benchmark, publish, timestamp, or otherwise freeze the public manifest. Do not reveal the private seeds.

## Run

Use the private bank directly:

```bash
python -m fca_bench.run --config config.qwen3_8b.example.json \
  --suite all --seed-bank sealed_seed_bank.private.json \
  --output result.json
```

When `--seed-bank` is supplied, `--n` and `--seed` do not select instances. Every selected test must have an entry in the bank.

The result JSON records the SHA-256 hash of the supplied private bank.

## Reveal and verify

After results are frozen, reveal the private bank and run:

```bash
python -m fca_bench.sealing verify \
  --private sealed_seed_bank.private.json \
  --public sealed_seed_bank.public.json
```

A successful verification shows that the revealed seeds are the seeds committed before the run.

## Stronger contamination defense

For publication-scale evaluation, use all of the following:

- fresh random seeds;
- broad random supports for names, values, graphs, timelines, and causal structures;
- held-out generator templates not released before evaluation;
- deterministic objective scoring rather than an LLM judge whenever possible;
- versioned generator source;
- post-run disclosure of the private item bank.
